#include <stdlib.h>
#include "multibytecodec.h"


struct pypy_cjk_dec_s *pypy_cjk_dec_init(const MultibyteCodec *codec,
                                         char *inbuf, Py_ssize_t inlen)
{
  struct pypy_cjk_dec_s *d = malloc(sizeof(struct pypy_cjk_dec_s));
  if (!d)
    return NULL;
  if (codec->decinit != NULL && codec->decinit(&d->state, codec->config) != 0)
    goto errorexit;

  d->codec = codec;
  d->inbuf = inbuf;
  d->inbuf_end = inbuf + inlen;
  d->outbuf_start = malloc(inlen * sizeof(Py_UNICODE));
  if (!d->outbuf_start)
    goto errorexit;
  d->outbuf = d->outbuf_start;
  d->outbuf_end = d->outbuf_start + inlen;
  return d;

 errorexit:
  free(d);
  return NULL;
}

void pypy_cjk_dec_free(struct pypy_cjk_dec_s *d)
{
  free(d->outbuf_start);
  free(d);
}

long pypy_cjk_dec_chunk(struct pypy_cjk_dec_s *d)
{
  Py_ssize_t inleft = (Py_ssize_t)(d->inbuf_end - d->inbuf);
  Py_ssize_t outleft = (Py_ssize_t)(d->outbuf_end - d->outbuf);
  if (inleft == 0)
    return 0;
  return d->codec->decode(&d->state, d->codec->config,
                          &d->inbuf, inleft, &d->outbuf, outleft);
}

Py_UNICODE *pypy_cjk_dec_outbuf(struct pypy_cjk_dec_s *d)
{
  return d->outbuf_start;
}

Py_ssize_t pypy_cjk_dec_outlen(struct pypy_cjk_dec_s *d)
{
  return d->outbuf - d->outbuf_start;
}
