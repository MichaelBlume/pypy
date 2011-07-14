#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <string.h>

#include "example01.h"

//===========================================================================
payload::payload(double d) : m_data(d) {
    count++;
}
payload::payload(const payload& p) : m_data(p.m_data) {
    count++;
}
payload& payload::operator=(const payload& p) {
    if (this != &p) {
        m_data = p.m_data;
    }
    return *this;
}
payload::~payload() {
    count--;
}

double payload::getData() { return m_data; }
void payload::setData(double d) { m_data = d; }

// class-level data
int payload::count = 0;


//===========================================================================
example01::example01() : m_somedata(-99) {
    count++;
}
example01::example01(int a) : m_somedata(a) {
    count++;
}
example01::example01(const example01& e) : m_somedata(e.m_somedata) {
    count++;
}
example01& example01::operator=(const example01& e) {
    if (this != &e) {
        m_somedata = e.m_somedata;
    }
    return *this;
}
example01::~example01() {
    count--;
}

// class-level methods
int example01::staticAddOneToInt(int a) {
    return a + 1;
}
int example01::staticAddOneToInt(int a, int b) {
    return a + b + 1;
}
double example01::staticAddToDouble(double a) {
    return a + 0.01;
}
int example01::staticAtoi(const char* str) {
    return ::atoi(str);
}
char* example01::staticStrcpy(const char* strin) {
    char* strout = (char*)malloc(::strlen(strin)+1);
    ::strcpy(strout, strin);
    return strout;
}
void example01::staticSetPayload(payload* p, double d) {
    p->setData(d);
}

payload* example01::staticCyclePayload(payload* p, double d) {
    staticSetPayload(p, d);
    return p;
}

payload example01::staticCopyCyclePayload(payload* p, double d) {
    staticSetPayload(p, d);
    return *p;
}

int example01::getCount() {
    return count;
}

// instance methods
int example01::addDataToInt(int a) {
    return m_somedata + a;
}

double example01::addDataToDouble(double a) {
    return m_somedata + a;
}

int example01::addDataToAtoi(const char* str) {
    return ::atoi(str) + m_somedata;
}   

char* example01::addToStringValue(const char* str) {
    int out = ::atoi(str) + m_somedata;
    std::ostringstream ss;
    ss << out << std::ends;
    std::string result = ss.str();
    char* cresult = (char*)malloc(result.size()+1);
    ::strcpy(cresult, result.c_str());
    return cresult;
}

void example01::setPayload(payload* p) {
    p->setData(m_somedata);
}

payload* example01::cyclePayload(payload* p) {
    setPayload(p);
    return p;
}

payload example01::copyCyclePayload(payload* p) {
    setPayload(p);
    return *p;
}

// class-level data
int example01::count = 0;


// global
int globalAddOneToInt(int a) {
   return a + 1;
}

int ns_example01::globalAddOneToInt(int a) {
   return ::globalAddOneToInt(a);
}
