#include <string>

class payload {
public:
    payload(double d = 0.);
    payload(const payload& p);
    payload& operator=(const payload& e);
    ~payload();

    double getData();
    void setData(double d);

public:        // class-level data
    static int count;

private:
    double m_data;
};


class example01 {
public:
    example01();
    example01(int a);
    example01(const example01& e);
    example01& operator=(const example01& e);
    ~example01();

public:        // class-level methods
    static int staticAddOneToInt(int a);
    static int staticAddOneToInt(int a, int b);
    static double staticAddToDouble(double a);
    static int staticAtoi(const char* str);
    static char* staticStrcpy(const char* strin);
    static void staticSetPayload(payload* p, double d);
    static payload* staticCyclePayload(payload* p, double d);
    static payload staticCopyCyclePayload(payload* p, double d);
    static int getCount();

public:        // instance methods
    int addDataToInt(int a);
    double addDataToDouble(double a);
    int addDataToAtoi(const char* str);
    char* addToStringValue(const char* str);

    void setPayload(payload* p);
    payload* cyclePayload(payload* p);
    payload copyCyclePayload(payload* p);

public:        // class-level data
    static int count;

public:        // instance data
    int m_somedata;
};


// global functions
int globalAddOneToInt(int a);
namespace ns_example01 {
    int globalAddOneToInt(int a);
}

#define typeValue(itype, tname) \
   itype tname##Value(itype arg0, int argn=0, itype arg1=1, itype arg2=2)

// argument passing
class ArgPasser {        // use a class for now as methptrgetter not
public:                  // implemented for global functions
   typeValue(short, short);
   typeValue(unsigned short, ushort);
   typeValue(int, int);
   typeValue(unsigned int, uint);
   typeValue(long, long);
   typeValue(unsigned long, ulong);

   std::string stringValue(
      std::string arg0, int argn=0, std::string arg1 = "default");

   std::string stringRef(
      const std::string& arg0, int argn=0, const std::string& arg1="default");
};


// special case naming
class z_ {
public:
   z_& gime_z_(z_& z);
   int myint;
};
