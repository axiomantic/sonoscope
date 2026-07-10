// tools/clap_midi_host/clap_midi_host.c — sonoscope v2 production CLAP MIDI host.
//
// The C CLAP host implements the subprocess JSON
// contract with deterministic output. Evolved from a proven earlier spike: a minimal,
// SEMANTICALLY-DUMB in-process CLAP host
// that loads a .clap, drives a PLAYING clap_event_transport for a fixed window,
// and captures every note-OUTPUT event the plugin pushes — serializing RAW 3-byte
// MIDI + an absolute sample position. ALL semantic decoding (note_on/off, PPQ
// ticks, ordering) is Python's job (backend B1); this host never decides meaning.
//
// Contract (design §3):
//   stdin : ONE JSON object  { plugin_path, [plugin_id], transport{...},
//                              render{sample_rate,block_size}, [state_b64] }
//   stdout: ONE JSON object  { outcome:"success", events:[{t_samples,midi[3]}],
//                              meta{...} }        (exit 0)
//        or { outcome:"error", error:{code,message} }   (nonzero exit)
//
// Determinism (design §10): the output is a pure function of the input — no
// wall-clock, no rng. The absolute t_samples (block_start + event.time) makes the
// captured event set block-size invariant. Build with -Werror from pinned headers.
//
// Build: scripts/build_clap_midi_host.sh (pin-gates then compiles against
//        vendor/clap/include; output build/clap_midi_host).

#include <dlfcn.h>
#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <clap/clap.h>

// ============================================================================
// Error reporting — the ONLY stdout on failure is the single error object.
// A human-readable line also goes to stderr (design §3); exit is nonzero.
// stdout carries a success object ONLY after a clean full run (§3); nothing is
// ever printed to stdout mid-run, so a partial/garbage stream is impossible.
// ============================================================================

static void emit_error(const char *code, const char *message) {
  if (!code) code = "error";
  if (!message) message = "(no message)";
  // Escape the message for JSON (control chars, quote, backslash).
  fputs("{\"outcome\":\"error\",\"error\":{\"code\":\"", stdout);
  fputs(code, stdout);
  fputs("\",\"message\":\"", stdout);
  for (const char *p = message; *p; p++) {
    unsigned char c = (unsigned char)*p;
    if (c == '"' || c == '\\') {
      putchar('\\');
      putchar(c);
    } else if (c == '\n') {
      fputs("\\n", stdout);
    } else if (c == '\r') {
      fputs("\\r", stdout);
    } else if (c == '\t') {
      fputs("\\t", stdout);
    } else if (c < 0x20) {
      printf("\\u%04x", c);
    } else {
      putchar((int)c);
    }
  }
  fputs("\"}}\n", stdout);
  fflush(stdout);
  fprintf(stderr, "[clap_midi_host] ERROR %s: %s\n", code, message);
}

// FAIL: emit the error object, clean up nothing (caller owns cleanup), return 1.
#define FAIL(code, msg)                                                                            \
  do {                                                                                             \
    emit_error((code), (msg));                                                                     \
    goto fail;                                                                                     \
  } while (0)

// ============================================================================
// Minimal JSON parser (recursive descent -> tagged DOM).
// Deliberately small; parses the fixed request shape robustly (nested objects,
// arrays, strings with escapes, numbers, bool, null). No external dependency.
// ============================================================================

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype_t;

typedef struct jnode {
  jtype_t type;
  bool bval;         // J_BOOL
  double num;        // J_NUM
  char *str;         // J_STR (NUL-terminated, unescaped)
  struct jnode **items; // J_ARR / J_OBJ values
  char **keys;          // J_OBJ keys (parallel to items)
  size_t count;         // J_ARR / J_OBJ length
} jnode_t;

typedef struct {
  const char *s;
  size_t n;
  size_t i;
  const char *err;
} jparser_t;

static jnode_t *jparse_value(jparser_t *p);

static void jskip_ws(jparser_t *p) {
  while (p->i < p->n) {
    char c = p->s[p->i];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
      p->i++;
    else
      break;
  }
}

static jnode_t *jnew(jtype_t t) {
  jnode_t *n = (jnode_t *)calloc(1, sizeof(jnode_t));
  if (n) n->type = t;
  return n;
}

static void jfree(jnode_t *n) {
  if (!n) return;
  if (n->type == J_STR) free(n->str);
  if (n->type == J_ARR || n->type == J_OBJ) {
    for (size_t k = 0; k < n->count; k++) {
      if (n->keys) free(n->keys[k]);
      jfree(n->items[k]);
    }
    free(n->items);
    free(n->keys);
  }
  free(n);
}

// Parse a JSON string literal starting at p->s[p->i] == '"'. Returns malloc'd,
// unescaped, NUL-terminated buffer. A unicode-zero escape would embed a NUL
// that strlen-based use later truncates; request strings never contain one.
static char *jparse_string_raw(jparser_t *p) {
  if (p->i >= p->n || p->s[p->i] != '"') {
    p->err = "expected string";
    return NULL;
  }
  p->i++; // opening quote
  size_t cap = 16, len = 0;
  char *out = (char *)malloc(cap);
  if (!out) {
    p->err = "oom";
    return NULL;
  }
  while (p->i < p->n) {
    char c = p->s[p->i++];
    if (c == '"') {
      out[len] = '\0';
      return out;
    }
    if (c == '\\') {
      if (p->i >= p->n) break;
      char e = p->s[p->i++];
      char decoded;
      switch (e) {
        case '"': decoded = '"'; break;
        case '\\': decoded = '\\'; break;
        case '/': decoded = '/'; break;
        case 'n': decoded = '\n'; break;
        case 't': decoded = '\t'; break;
        case 'r': decoded = '\r'; break;
        case 'b': decoded = '\b'; break;
        case 'f': decoded = '\f'; break;
        case 'u': {
          if (p->i + 4 > p->n) {
            p->err = "bad \\u escape";
            free(out);
            return NULL;
          }
          unsigned int cp = 0;
          for (int k = 0; k < 4; k++) {
            char h = p->s[p->i++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
            else {
              p->err = "bad hex in \\u";
              free(out);
              return NULL;
            }
          }
          // Minimal UTF-8 encode of the BMP code point (request paths are ASCII
          // in practice; this keeps the parser correct for any valid input).
          if (cp < 0x80) {
            decoded = (char)cp;
          } else {
            // Grow + emit multi-byte sequence, then continue the loop.
            char buf[3];
            int bl;
            if (cp < 0x800) {
              buf[0] = (char)(0xC0 | (cp >> 6));
              buf[1] = (char)(0x80 | (cp & 0x3F));
              bl = 2;
            } else {
              buf[0] = (char)(0xE0 | (cp >> 12));
              buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
              buf[2] = (char)(0x80 | (cp & 0x3F));
              bl = 3;
            }
            for (int bi = 0; bi < bl; bi++) {
              if (len + 1 >= cap) {
                cap *= 2;
                char *grown = (char *)realloc(out, cap);
                if (!grown) {
                  p->err = "oom";
                  free(out);
                  return NULL;
                }
                out = grown;
              }
              out[len++] = buf[bi];
            }
            continue;
          }
          break;
        }
        default:
          p->err = "bad escape";
          free(out);
          return NULL;
      }
      if (len + 1 >= cap) {
        cap *= 2;
        char *grown = (char *)realloc(out, cap);
        if (!grown) {
          p->err = "oom";
          free(out);
          return NULL;
        }
        out = grown;
      }
      out[len++] = decoded;
    } else {
      if (len + 1 >= cap) {
        cap *= 2;
        char *grown = (char *)realloc(out, cap);
        if (!grown) {
          p->err = "oom";
          free(out);
          return NULL;
        }
        out = grown;
      }
      out[len++] = c;
    }
  }
  p->err = "unterminated string";
  free(out);
  return NULL;
}

static bool jappend(jnode_t *parent, char *key, jnode_t *val) {
  jnode_t **ni = (jnode_t **)realloc(parent->items, (parent->count + 1) * sizeof(jnode_t *));
  if (!ni) return false;
  parent->items = ni;
  if (parent->type == J_OBJ) {
    char **nk = (char **)realloc(parent->keys, (parent->count + 1) * sizeof(char *));
    if (!nk) return false;
    parent->keys = nk;
    parent->keys[parent->count] = key;
  }
  parent->items[parent->count] = val;
  parent->count++;
  return true;
}

static jnode_t *jparse_value(jparser_t *p) {
  jskip_ws(p);
  if (p->i >= p->n) {
    p->err = "unexpected end of input";
    return NULL;
  }
  char c = p->s[p->i];
  if (c == '"') {
    char *str = jparse_string_raw(p);
    if (!str) return NULL;
    jnode_t *n = jnew(J_STR);
    if (!n) {
      free(str);
      p->err = "oom";
      return NULL;
    }
    n->str = str;
    return n;
  }
  if (c == '{') {
    p->i++;
    jnode_t *n = jnew(J_OBJ);
    if (!n) {
      p->err = "oom";
      return NULL;
    }
    jskip_ws(p);
    if (p->i < p->n && p->s[p->i] == '}') {
      p->i++;
      return n;
    }
    for (;;) {
      jskip_ws(p);
      char *key = jparse_string_raw(p);
      if (!key) {
        jfree(n);
        return NULL;
      }
      jskip_ws(p);
      if (p->i >= p->n || p->s[p->i] != ':') {
        p->err = "expected ':'";
        free(key);
        jfree(n);
        return NULL;
      }
      p->i++;
      jnode_t *val = jparse_value(p);
      if (!val) {
        free(key);
        jfree(n);
        return NULL;
      }
      if (!jappend(n, key, val)) {
        p->err = "oom";
        free(key);
        jfree(val);
        jfree(n);
        return NULL;
      }
      jskip_ws(p);
      if (p->i < p->n && p->s[p->i] == ',') {
        p->i++;
        continue;
      }
      if (p->i < p->n && p->s[p->i] == '}') {
        p->i++;
        return n;
      }
      p->err = "expected ',' or '}'";
      jfree(n);
      return NULL;
    }
  }
  if (c == '[') {
    p->i++;
    jnode_t *n = jnew(J_ARR);
    if (!n) {
      p->err = "oom";
      return NULL;
    }
    jskip_ws(p);
    if (p->i < p->n && p->s[p->i] == ']') {
      p->i++;
      return n;
    }
    for (;;) {
      jnode_t *val = jparse_value(p);
      if (!val) {
        jfree(n);
        return NULL;
      }
      if (!jappend(n, NULL, val)) {
        p->err = "oom";
        jfree(val);
        jfree(n);
        return NULL;
      }
      jskip_ws(p);
      if (p->i < p->n && p->s[p->i] == ',') {
        p->i++;
        continue;
      }
      if (p->i < p->n && p->s[p->i] == ']') {
        p->i++;
        return n;
      }
      p->err = "expected ',' or ']'";
      jfree(n);
      return NULL;
    }
  }
  if (c == 't' || c == 'f') {
    const char *lit = (c == 't') ? "true" : "false";
    size_t ll = (c == 't') ? 4 : 5;
    if (p->i + ll <= p->n && strncmp(p->s + p->i, lit, ll) == 0) {
      p->i += ll;
      jnode_t *n = jnew(J_BOOL);
      if (!n) {
        p->err = "oom";
        return NULL;
      }
      n->bval = (c == 't');
      return n;
    }
    p->err = "invalid literal";
    return NULL;
  }
  if (c == 'n') {
    if (p->i + 4 <= p->n && strncmp(p->s + p->i, "null", 4) == 0) {
      p->i += 4;
      return jnew(J_NULL);
    }
    p->err = "invalid literal";
    return NULL;
  }
  // Number.
  {
    size_t start = p->i;
    if (p->s[p->i] == '-') p->i++;
    bool any = false;
    while (p->i < p->n) {
      char d = p->s[p->i];
      if ((d >= '0' && d <= '9') || d == '.' || d == 'e' || d == 'E' || d == '+' || d == '-') {
        p->i++;
        any = true;
      } else {
        break;
      }
    }
    if (!any) {
      p->err = "unexpected character";
      return NULL;
    }
    char tmp[64];
    size_t len = p->i - start;
    if (len >= sizeof(tmp)) {
      p->err = "number too long";
      return NULL;
    }
    memcpy(tmp, p->s + start, len);
    tmp[len] = '\0';
    // Reject malformed literals: strtod parses only a valid prefix and silently
    // ignores trailing garbage, so "1.2.3", "1e", "--5", "1..2", "1e+" would
    // otherwise be accepted. Require strtod to consume the ENTIRE collected
    // token — end must point at the terminating NUL (tmp + len).
    char *end = NULL;
    double val = strtod(tmp, &end);
    if (end != tmp + len) {
      p->err = "invalid number";
      return NULL;
    }
    jnode_t *n = jnew(J_NUM);
    if (!n) {
      p->err = "oom";
      return NULL;
    }
    n->num = val;
    return n;
  }
}

static jnode_t *jparse(const char *s, size_t n, const char **err) {
  jparser_t p = {s, n, 0, NULL};
  jnode_t *root = jparse_value(&p);
  if (!root) {
    *err = p.err ? p.err : "parse error";
    return NULL;
  }
  jskip_ws(&p);
  if (p.i != p.n) {
    // Trailing garbage after the single object.
    jfree(root);
    *err = "trailing content after JSON value";
    return NULL;
  }
  *err = NULL;
  return root;
}

// --- DOM accessors ---------------------------------------------------------

static jnode_t *jobj_get(const jnode_t *o, const char *key) {
  if (!o || o->type != J_OBJ) return NULL;
  for (size_t k = 0; k < o->count; k++) {
    if (o->keys[k] && strcmp(o->keys[k], key) == 0) return o->items[k];
  }
  return NULL;
}

// Fetch a required number field from an object. Returns false (and sets *found
// = whether the key existed) on absence/type-mismatch.
static bool jget_num(const jnode_t *o, const char *key, double *out) {
  jnode_t *v = jobj_get(o, key);
  if (!v || v->type != J_NUM) return false;
  *out = v->num;
  return true;
}

static const char *jget_str(const jnode_t *o, const char *key) {
  jnode_t *v = jobj_get(o, key);
  if (!v || v->type != J_STR) return NULL;
  return v->str;
}

// ============================================================================
// base64 decode (state_b64, design §3 state-blob load path — plumbed, no-op
// until a blob is supplied). Standard alphabet, '=' padding. Returns malloc'd
// buffer + length; NULL on any invalid character.
// ============================================================================

static int b64val(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

static unsigned char *b64decode(const char *in, size_t *out_len) {
  size_t in_len = strlen(in);
  // Ignore whitespace by pre-counting significant chars.
  unsigned char *out = (unsigned char *)malloc(in_len / 4 * 3 + 4);
  if (!out) return NULL;
  size_t olen = 0;
  int quad[4];
  int qn = 0;
  for (size_t i = 0; i < in_len; i++) {
    char c = in[i];
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    if (c == '=') {
      // Padding: flush whatever we have.
      quad[qn++] = -2;
    } else {
      int v = b64val(c);
      if (v < 0) {
        free(out);
        return NULL;
      }
      quad[qn++] = v;
    }
    if (qn == 4) {
      int a = quad[0], b = quad[1], cc = quad[2], d = quad[3];
      if (a < 0 || b < 0) {
        free(out);
        return NULL;
      }
      out[olen++] = (unsigned char)((a << 2) | (b >> 4));
      if (cc != -2) out[olen++] = (unsigned char)(((b & 0x0F) << 4) | (cc >> 2));
      if (cc != -2 && d != -2) out[olen++] = (unsigned char)(((cc & 0x03) << 6) | d);
      qn = 0;
    }
  }
  if (qn != 0) {
    // Truncated final quad without padding.
    free(out);
    return NULL;
  }
  *out_len = olen;
  return out;
}

// clap_istream backed by an in-memory buffer (for state.load).
typedef struct {
  const unsigned char *data;
  size_t len;
  size_t pos;
} mem_istream_ctx_t;

static int64_t mem_istream_read(const struct clap_istream *stream, void *buffer, uint64_t size) {
  mem_istream_ctx_t *ctx = (mem_istream_ctx_t *)stream->ctx;
  size_t remaining = ctx->len - ctx->pos;
  size_t n = (size < remaining) ? (size_t)size : remaining;
  if (n > 0) {
    memcpy(buffer, ctx->data + ctx->pos, n);
    ctx->pos += n;
  }
  return (int64_t)n; // 0 == EOF
}

// ============================================================================
// Captured-event store — RAW 3-byte MIDI + absolute sample position ONLY.
// No note_on/off classification here (semantically dumb; Python decodes).
// ============================================================================

#define MAX_CAP 65536
typedef struct {
  uint64_t t_samples; // absolute: block_start + event.time
  uint8_t midi[3];    // status, data1, data2
} cap_event_t;

static cap_event_t g_cap[MAX_CAP];
static uint32_t g_cap_n = 0;
static uint64_t g_block_start = 0; // absolute sample offset of the block in flight

static void cap_push_raw(uint64_t t, uint8_t b0, uint8_t b1, uint8_t b2) {
  if (g_cap_n >= MAX_CAP) return; // saturate silently; overflow reported at end
  g_cap[g_cap_n].t_samples = t;
  g_cap[g_cap_n].midi[0] = b0;
  g_cap[g_cap_n].midi[1] = b1;
  g_cap[g_cap_n].midi[2] = b2;
  g_cap_n++;
}

static uint8_t vel_to_byte(double velocity) {
  double v = round(velocity * 127.0);
  if (v < 0.0) v = 0.0;
  if (v > 127.0) v = 127.0;
  return (uint8_t)v;
}

// out_events.try_push — the plugin pushes note-OUTPUT events here. Capture core-
// space MIDI (raw) and CLAP NOTE_ON/NOTE_OFF (normalized to raw 3 bytes). Reference Sequencer
// emits the MIDI dialect; the NOTE normalization keeps the host correct for a
// future plugin on the CLAP note dialect (design §3 "and CLAP note events,
// normalized to raw"). All other event types are ignored (not MIDI emission).
static bool out_events_try_push(const clap_output_events_t *l, const clap_event_header_t *e) {
  (void)l;
  if (!e) return true;
  if (e->space_id != CLAP_CORE_EVENT_SPACE_ID) return true;
  if (e->type == CLAP_EVENT_MIDI) {
    const clap_event_midi_t *m = (const clap_event_midi_t *)e;
    cap_push_raw(g_block_start + e->time, m->data[0], m->data[1], m->data[2]);
  } else if (e->type == CLAP_EVENT_NOTE_ON) {
    const clap_event_note_t *n = (const clap_event_note_t *)e;
    cap_push_raw(g_block_start + e->time, (uint8_t)(0x90 | (n->channel & 0x0F)),
                 (uint8_t)(n->key & 0x7F), vel_to_byte(n->velocity));
  } else if (e->type == CLAP_EVENT_NOTE_OFF) {
    const clap_event_note_t *n = (const clap_event_note_t *)e;
    cap_push_raw(g_block_start + e->time, (uint8_t)(0x80 | (n->channel & 0x0F)),
                 (uint8_t)(n->key & 0x7F), vel_to_byte(n->velocity));
  }
  return true;
}

// --- host / event-queue vtables --------------------------------------------

static const void *host_get_extension(const clap_host_t *host, const char *ext_id) {
  (void)host;
  (void)ext_id;
  return NULL;
}
static void host_request_restart(const clap_host_t *host) { (void)host; }
static void host_request_process(const clap_host_t *host) { (void)host; }
static void host_request_callback(const clap_host_t *host) { (void)host; }

static uint32_t in_events_size(const clap_input_events_t *l) {
  (void)l;
  return 0;
}
static const clap_event_header_t *in_events_get(const clap_input_events_t *l, uint32_t i) {
  (void)l;
  (void)i;
  return NULL;
}

// ============================================================================
// stdin slurp
// ============================================================================

static char *read_all_stdin(size_t *out_len) {
  size_t cap = 65536, len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) return NULL;
  for (;;) {
    if (len == cap) {
      cap *= 2;
      char *grown = (char *)realloc(buf, cap);
      if (!grown) {
        free(buf);
        return NULL;
      }
      buf = grown;
    }
    size_t got = fread(buf + len, 1, cap - len, stdin);
    len += got;
    if (got == 0) {
      if (feof(stdin)) break;
      if (ferror(stdin)) {
        free(buf);
        return NULL;
      }
    }
  }
  *out_len = len;
  return buf;
}

// ============================================================================
// main
// ============================================================================

int main(void) {
  // Force the C locale for numeric parsing: strtod (below) is locale-dependent,
  // so a comma-decimal locale (LC_NUMERIC=de_DE etc.) would fail to parse JSON
  // floats like "120.0". Pin LC_NUMERIC to "C" before ANY parsing so the JSON
  // number decode is locale-invariant and deterministic.
  setlocale(LC_NUMERIC, "C");

  int rc = 1;

  // Resources (freed at fail: label).
  char *input = NULL;
  jnode_t *root = NULL;
  unsigned char *state_blob = NULL;
  void *dl = NULL;
  const clap_plugin_entry_t *entry = NULL;
  const clap_plugin_t *plugin = NULL;
  bool entry_inited = false;
  bool activated = false;
  bool processing = false;

  // -------- 1. read + parse the request JSON (design §3 input) -------------
  size_t input_len = 0;
  input = read_all_stdin(&input_len);
  if (!input) FAIL("bad_input", "failed to read stdin");
  if (input_len == 0) FAIL("bad_input", "empty stdin (expected one JSON object)");

  const char *jerr = NULL;
  root = jparse(input, input_len, &jerr);
  if (!root) FAIL("bad_input", jerr ? jerr : "malformed JSON");
  if (root->type != J_OBJ) FAIL("bad_input", "top-level JSON is not an object");

  // plugin_path (required).
  const char *plugin_path = jget_str(root, "plugin_path");
  if (!plugin_path || plugin_path[0] == '\0')
    FAIL("bad_input", "missing required string field 'plugin_path'");

  // plugin_id (optional).
  const char *want_id = jget_str(root, "plugin_id");

  // render.{sample_rate,block_size} (required, > 0).
  const jnode_t *render = jobj_get(root, "render");
  if (!render || render->type != J_OBJ) FAIL("bad_input", "missing required object 'render'");
  double d_sr = 0.0, d_bs = 0.0;
  if (!jget_num(render, "sample_rate", &d_sr))
    FAIL("bad_input", "missing required number 'render.sample_rate'");
  if (!jget_num(render, "block_size", &d_bs))
    FAIL("bad_input", "missing required number 'render.block_size'");
  if (d_sr <= 0.0) FAIL("bad_input", "'render.sample_rate' must be > 0");
  if (d_bs < 1.0) FAIL("bad_input", "'render.block_size' must be >= 1");
  const double sample_rate = d_sr;
  const uint32_t block_size = (uint32_t)d_bs;

  // transport.{tempo_bpm,start_position_beats,duration_beats,tsig_num,tsig_den}.
  // 'playing' is optional (default true); H1 drives PLAYING. A false 'playing'
  // clears IS_PLAYING so a play-gated plugin emits nothing (plan H1 acceptance:
  // "playing:false -> 0 events"). H1's demonstration always uses playing=true.
  const jnode_t *transport = jobj_get(root, "transport");
  if (!transport || transport->type != J_OBJ)
    FAIL("bad_input", "missing required object 'transport'");
  double tempo_bpm = 0.0, start_beats = 0.0, duration_beats = 0.0, d_tnum = 0.0, d_tden = 0.0;
  if (!jget_num(transport, "tempo_bpm", &tempo_bpm))
    FAIL("bad_input", "missing required number 'transport.tempo_bpm'");
  if (!jget_num(transport, "start_position_beats", &start_beats))
    FAIL("bad_input", "missing required number 'transport.start_position_beats'");
  if (!jget_num(transport, "duration_beats", &duration_beats))
    FAIL("bad_input", "missing required number 'transport.duration_beats'");
  if (!jget_num(transport, "tsig_num", &d_tnum))
    FAIL("bad_input", "missing required number 'transport.tsig_num'");
  if (!jget_num(transport, "tsig_den", &d_tden))
    FAIL("bad_input", "missing required number 'transport.tsig_den'");
  if (tempo_bpm <= 0.0) FAIL("bad_input", "'transport.tempo_bpm' must be > 0");
  if (duration_beats < 0.0) FAIL("bad_input", "'transport.duration_beats' must be >= 0");
  if (d_tnum < 1.0 || d_tden < 1.0) FAIL("bad_input", "time signature fields must be >= 1");
  const uint16_t tsig_num = (uint16_t)d_tnum;
  const uint16_t tsig_den = (uint16_t)d_tden;
  bool playing = true;
  {
    jnode_t *pv = jobj_get(transport, "playing");
    if (pv) {
      if (pv->type != J_BOOL) FAIL("bad_input", "'transport.playing' must be a boolean");
      playing = pv->bval;
    }
  }

  // state_b64 (optional, POST-M4 CLAP state blob). Decoded now; loaded after
  // activate, before start_processing.
  size_t state_len = 0;
  const char *state_b64 = jget_str(root, "state_b64");
  if (state_b64 && state_b64[0] != '\0') {
    state_blob = b64decode(state_b64, &state_len);
    if (!state_blob) FAIL("state_decode_failed", "'state_b64' is not valid base64");
  }

  // -------- 2. load the .clap and instantiate the plugin -------------------
  // dlopen resolves the bundle binary (caller passes the inner Contents/MacOS
  // binary or a bare .so/.dylib; bundle-path resolution is the backend's job).
  dl = dlopen(plugin_path, RTLD_LOCAL | RTLD_NOW);
  if (!dl) {
    // dlerror() clears its state on read AND may return NULL — capture it once
    // into a stable buffer so the error path never dereferences NULL.
    const char *dlerr = dlerror();
    FAIL("dlopen_failed", dlerr ? dlerr : "dlopen returned NULL");
  }

  entry = (const clap_plugin_entry_t *)dlsym(dl, "clap_entry");
  if (!entry) FAIL("no_clap_entry", "symbol 'clap_entry' not found in plugin binary");
  if (!entry->init(plugin_path)) FAIL("entry_init_failed", "clap_entry.init returned false");
  entry_inited = true;

  const clap_plugin_factory_t *factory =
      (const clap_plugin_factory_t *)entry->get_factory(CLAP_PLUGIN_FACTORY_ID);
  if (!factory) FAIL("no_factory", "get_factory(clap.plugin-factory) returned NULL");
  uint32_t count = factory->get_plugin_count(factory);
  if (count < 1) FAIL("no_plugins", "plugin factory is empty");

  // Select the descriptor: by plugin_id if supplied, else index 0.
  const clap_plugin_descriptor_t *desc = NULL;
  if (want_id) {
    for (uint32_t i = 0; i < count; i++) {
      const clap_plugin_descriptor_t *d = factory->get_plugin_descriptor(factory, i);
      if (d && d->id && strcmp(d->id, want_id) == 0) {
        desc = d;
        break;
      }
    }
    if (!desc) FAIL("plugin_not_found", "no plugin in factory matches the requested plugin_id");
  } else {
    desc = factory->get_plugin_descriptor(factory, 0);
    if (!desc) FAIL("plugin_not_found", "get_plugin_descriptor(0) returned NULL");
  }

  clap_host_t host = {0};
  host.clap_version = (clap_version_t)CLAP_VERSION_INIT;
  host.host_data = NULL;
  host.name = "sonoscope-clap-midi-host";
  host.vendor = "sonoscope";
  host.url = "";
  host.version = "1";
  host.get_extension = host_get_extension;
  host.request_restart = host_request_restart;
  host.request_process = host_request_process;
  host.request_callback = host_request_callback;

  plugin = factory->create_plugin(factory, &host, desc->id);
  if (!plugin) FAIL("create_plugin_failed", "factory.create_plugin returned NULL");
  if (!plugin->init(plugin)) FAIL("plugin_init_failed", "plugin.init returned false");

  // Note-port info for meta (informational; capture is port-agnostic).
  uint32_t note_in = 0, note_out = 0, out_dialects = 0;
  const clap_plugin_note_ports_t *np =
      (const clap_plugin_note_ports_t *)plugin->get_extension(plugin, CLAP_EXT_NOTE_PORTS);
  if (np) {
    note_in = np->count(plugin, true);
    note_out = np->count(plugin, false);
    if (note_out > 0) {
      clap_note_port_info_t info;
      memset(&info, 0, sizeof(info));
      if (np->get(plugin, 0, false, &info)) out_dialects = info.supported_dialects;
    }
  }

  if (!plugin->activate(plugin, sample_rate, 1, block_size))
    FAIL("activate_failed", "plugin.activate returned false");
  activated = true;

  // -------- 3. load state (POST-M4; no-op when no blob supplied) ------------
  if (state_blob) {
    const clap_plugin_state_t *state_ext =
        (const clap_plugin_state_t *)plugin->get_extension(plugin, CLAP_EXT_STATE);
    if (!state_ext || !state_ext->load)
      FAIL("state_load_failed", "plugin does not implement the clap.state extension");
    mem_istream_ctx_t ictx = {state_blob, state_len, 0};
    clap_istream_t istream = {0};
    istream.ctx = &ictx;
    istream.read = mem_istream_read;
    if (!state_ext->load(plugin, &istream))
      FAIL("state_load_failed", "plugin.state.load returned false");
  }

  if (!plugin->start_processing(plugin))
    FAIL("start_processing_failed", "plugin.start_processing returned false");
  processing = true;

  // -------- 4. drive the transport, capture note-output --------------------
  // total_frames = duration_beats * (60/tempo) seconds * sample_rate, rounded.
  const uint64_t total_frames =
      (uint64_t)llround(duration_beats * 60.0 / tempo_bpm * sample_rate);
  const double start_sec = start_beats * 60.0 / tempo_bpm;
  const double beats_per_bar = (double)tsig_num; // 4/4-style; informational bar math

  // Dummy stereo audio output — Reference Sequencer reports 1 audio out and writes silence.
  float *chL = (float *)calloc(block_size, sizeof(float));
  float *chR = (float *)calloc(block_size, sizeof(float));
  if (!chL || !chR) {
    free(chL);
    free(chR);
    FAIL("bad_input", "failed to allocate audio scratch buffers");
  }
  float *chans[2] = {chL, chR};
  clap_audio_buffer_t audio_out = {0};
  audio_out.data32 = chans;
  audio_out.channel_count = 2;

  clap_input_events_t in_events = {0};
  in_events.size = in_events_size;
  in_events.get = in_events_get;
  clap_output_events_t out_events = {0};
  out_events.try_push = out_events_try_push;

  // Render FULL blocks extending one block PAST the capture window so a full
  // block always spans total_frames. A boundary event whose true position is
  // exactly total_frames (the loop downbeat) then lands at its TRUE in-block
  // offset inside a full block instead of being clamped inward by a TRUNCATED
  // final block — the capture-boundary artifact. The headroom is filtered back
  // to the true half-open window [0, total_frames) before serialize (below).
  const uint64_t render_frames = (total_frames / block_size + 1) * block_size;
  bool process_error = false;
  uint64_t steady = 0;
  while (steady < render_frames) {
    uint32_t nframes = block_size; // always a FULL block (no final-block shrink)
    g_block_start = steady;

    // Song position advances from start_position_beats; block-size independent.
    const double cur_sec = start_sec + (double)steady / sample_rate;
    const double cur_beats = cur_sec * tempo_bpm / 60.0;
    const double bar_start_beats = floor(cur_beats / beats_per_bar) * beats_per_bar;

    clap_event_transport_t tr = {0};
    tr.header.size = sizeof(tr);
    tr.header.time = 0;
    tr.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
    tr.header.type = CLAP_EVENT_TRANSPORT;
    tr.header.flags = 0;
    tr.flags = CLAP_TRANSPORT_HAS_TEMPO | CLAP_TRANSPORT_HAS_BEATS_TIMELINE |
               CLAP_TRANSPORT_HAS_SECONDS_TIMELINE | CLAP_TRANSPORT_HAS_TIME_SIGNATURE |
               (playing ? (uint32_t)CLAP_TRANSPORT_IS_PLAYING : 0u);
    tr.song_pos_beats = (clap_beattime)llround(cur_beats * CLAP_BEATTIME_FACTOR);
    tr.song_pos_seconds = (clap_sectime)llround(cur_sec * CLAP_SECTIME_FACTOR);
    tr.tempo = tempo_bpm;
    tr.tempo_inc = 0.0;
    tr.bar_start = (clap_beattime)llround(bar_start_beats * CLAP_BEATTIME_FACTOR);
    tr.bar_number = (int32_t)llround(bar_start_beats / beats_per_bar);
    tr.tsig_num = tsig_num;
    tr.tsig_denom = tsig_den;

    for (uint32_t f = 0; f < block_size; f++) {
      chL[f] = 0.0f;
      chR[f] = 0.0f;
    }

    clap_process_t proc = {0};
    proc.steady_time = (int64_t)steady;
    proc.frames_count = nframes;
    proc.transport = &tr;
    proc.audio_inputs = NULL;
    proc.audio_inputs_count = 0;
    proc.audio_outputs = &audio_out;
    proc.audio_outputs_count = 1;
    proc.in_events = &in_events;
    proc.out_events = &out_events;

    clap_process_status st = plugin->process(plugin, &proc);
    if (st == CLAP_PROCESS_ERROR) {
      process_error = true;
      break;
    }
    steady += nframes;
  }

  free(chL);
  free(chR);
  if (process_error) FAIL("process_error", "plugin.process returned CLAP_PROCESS_ERROR");
  if (g_cap_n >= MAX_CAP)
    FAIL("capture_overflow", "captured event count hit MAX_CAP; capture is incomplete");

  // Filter the captured events back to the TRUE half-open window
  // [0, total_frames): drop the boundary/headroom events the one-block render
  // past the window surfaced. The host output contract (events within
  // [0, total_frames), meta.duration_samples = total_frames, n_events =
  // in-window count) is UNCHANGED — this only excludes events at/after the
  // window boundary that the extended full-block render emitted at their true
  // positions, instead of clamping the boundary event inward. In-window event
  // positions are untouched (t_samples is absolute).
  {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < g_cap_n; i++) {
      if (g_cap[i].t_samples < total_frames) g_cap[kept++] = g_cap[i];
    }
    g_cap_n = kept;
  }

  // -------- 5. serialize the success object (design §3 output) -------------
  // Emitted exactly once, only after a clean full run. All numbers are integers
  // or fixed strings, so the output is byte-identical across identical inputs and
  // the events array is block-size invariant (t_samples is absolute).
  fputs("{\"outcome\":\"success\",\"events\":[", stdout);
  for (uint32_t i = 0; i < g_cap_n; i++) {
    if (i) putchar(',');
    printf("{\"t_samples\":%llu,\"midi\":[%u,%u,%u]}", (unsigned long long)g_cap[i].t_samples,
           g_cap[i].midi[0], g_cap[i].midi[1], g_cap[i].midi[2]);
  }
  fputs("],\"meta\":{", stdout);
  printf("\"sample_rate\":%d,", (int)sample_rate);
  printf("\"block_size\":%u,", block_size);
  printf("\"duration_samples\":%llu,", (unsigned long long)total_frames);
  fputs("\"plugin_id\":\"", stdout);
  for (const char *p = desc->id ? desc->id : ""; *p; p++) {
    if (*p == '"' || *p == '\\') putchar('\\');
    putchar(*p);
  }
  fputs("\",\"plugin_name\":\"", stdout);
  for (const char *p = desc->name ? desc->name : ""; *p; p++) {
    if (*p == '"' || *p == '\\') putchar('\\');
    putchar(*p);
  }
  printf("\",\"n_events\":%u", g_cap_n);
  (void)note_in;
  (void)note_out;
  (void)out_dialects;
  fputs("}}\n", stdout);
  fflush(stdout);
  rc = 0;

fail:
  // -------- teardown (best effort; order mirrors the CLAP lifecycle) --------
  if (plugin) {
    if (processing) plugin->stop_processing(plugin);
    if (activated) plugin->deactivate(plugin);
    plugin->destroy(plugin);
  }
  if (entry_inited && entry) entry->deinit();
  if (dl) dlclose(dl);
  free(state_blob);
  jfree(root);
  free(input);
  return rc;
}
