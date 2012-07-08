#include <stdio.h>
#include <stdarg.h>
#include <errno.h>

static char _testfunc0(char a, char b)
{
    return a + b;
}
static long _testfunc1(int a, long b)
{
    return (long)a + b;
}
static long long _testfunc2(long long a, long long b)
{
    return a + b;
}
static double _testfunc3(float a, double b)
{
    return a + b;
}
static float _testfunc4(float a, double b)
{
    return (float)(a + b);
}
static void _testfunc5(void)
{
    errno = errno + 15;
}
static int *_testfunc6(int *x)
{
    static int y;
    y = *x - 1000;
    return &y;
}
struct _testfunc7_s { unsigned char a1; short a2; };
static short _testfunc7(struct _testfunc7_s inlined)
{
    return inlined.a1 + inlined.a2;
}
static int _testfunc9(int num, ...)
{
    va_list vargs;
    int i, total = 0;
    va_start(vargs, num);
    for (i=0; i<num; i++) {
        int value = va_arg(vargs, int);
        if (value == 0)
            value = -66666666;
        total += value;
    }
    va_end(vargs);
    return total;
}

static struct _testfunc7_s _testfunc10(int n)
{
    struct _testfunc7_s result;
    result.a1 = n;
    result.a2 = n * n;
    return result;
}

struct _testfunc11_s { int a1, a2; };
static struct _testfunc11_s _testfunc11(int n)
{
    struct _testfunc11_s result;
    result.a1 = n;
    result.a2 = n * n;
    return result;
}

struct _testfunc12_s { double a1; };
static struct _testfunc12_s _testfunc12(int n)
{
    struct _testfunc12_s result;
    result.a1 = n;
    return result;
}

struct _testfunc13_s { int a1, a2, a3; };
static struct _testfunc13_s _testfunc13(int n)
{
    struct _testfunc13_s result;
    result.a1 = n;
    result.a2 = n * n;
    result.a3 = n * n * n;
    return result;
}

struct _testfunc14_s { float a1; };
static struct _testfunc14_s _testfunc14(int n)
{
    struct _testfunc14_s result;
    result.a1 = (float)n;
    return result;
}

struct _testfunc15_s { float a1; int a2; };
static struct _testfunc15_s _testfunc15(int n)
{
    struct _testfunc15_s result;
    result.a1 = (float)n;
    result.a2 = n * n;
    return result;
}

struct _testfunc16_s { float a1, a2; };
static struct _testfunc16_s _testfunc16(int n)
{
    struct _testfunc16_s result;
    result.a1 = (float)n;
    result.a2 = -(float)n;
    return result;
}

struct _testfunc17_s { int a1; float a2; };
static struct _testfunc17_s _testfunc17(int n)
{
    struct _testfunc17_s result;
    result.a1 = n;
    result.a2 = (float)n * (float)n;
    return result;
}

void *gettestfunc(int num)
{
    void *f;
    switch (num) {
    case 0: f = &_testfunc0; break;
    case 1: f = &_testfunc1; break;
    case 2: f = &_testfunc2; break;
    case 3: f = &_testfunc3; break;
    case 4: f = &_testfunc4; break;
    case 5: f = &_testfunc5; break;
    case 6: f = &_testfunc6; break;
    case 7: f = &_testfunc7; break;
    case 8: f = stderr; break;
    case 9: f = &_testfunc9; break;
    case 10: f = &_testfunc10; break;
    case 11: f = &_testfunc11; break;
    case 12: f = &_testfunc12; break;
    case 13: f = &_testfunc13; break;
    case 14: f = &_testfunc14; break;
    case 15: f = &_testfunc15; break;
    case 16: f = &_testfunc16; break;
    case 17: f = &_testfunc17; break;
    default:
        return NULL;
    }
    return f;
}
