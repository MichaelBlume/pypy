#include "cppyy.h"
#include "cintcwrapper.h"

#include "Api.h"

#include "TList.h"

#include "TBaseClass.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassRef.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TMethodArg.h"

#include <string.h>
#include <iostream>
#include <map>
#include <string>
#include <utility>


/* data for life time management ------------------------------------------ */
typedef std::vector<TClassRef> ClassRefs_t;
static ClassRefs_t g_classrefs(1);

typedef std::map<std::string, ClassRefs_t::size_type> ClassRefIndices_t;
static ClassRefIndices_t g_classref_indices;


/* local helpers ---------------------------------------------------------- */
static inline char* cppstring_to_cstring(const std::string& name) {
    char* name_char = (char*)malloc(name.size() + 1);
    strcpy(name_char, name.c_str());
    return name_char;
}

static inline char* type_cppstring_to_cstring(const std::string& tname) {
    G__TypeInfo ti(tname.c_str());
    std::string name = ti.IsValid() ? ti.TrueName() : tname;
    return cppstring_to_cstring(name);
}

static inline TClassRef type_from_handle(cppyy_typehandle_t handle) {
    return g_classrefs[(ClassRefs_t::size_type)handle];
}


/* name to handle --------------------------------------------------------- */
cppyy_typehandle_t cppyy_get_typehandle(const char* class_name) {
    ClassRefIndices_t::iterator icr = g_classref_indices.find(class_name);
    if (icr != g_classref_indices.end())
       return (cppyy_typehandle_t)icr->second;

    ClassRefs_t::size_type sz = g_classrefs.size();
    g_classref_indices[class_name] = sz;
    g_classrefs.push_back(TClassRef(class_name));
    return (cppyy_typehandle_t)sz;
}

cppyy_typehandle_t cppyy_get_templatehandle(const char* template_name) {
   return cppyy_get_typehandle(template_name);
}


/* memory management ------------------------------------------------------ */
void* cppyy_allocate(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    return malloc(cr->Size());
}

void cppyy_deallocate(cppyy_typehandle_t /*handle*/, cppyy_object_t instance) {
    free((void*)instance);
}

void cppyy_destruct(cppyy_typehandle_t handle, cppyy_object_t self) {
    TClassRef cr = type_from_handle(handle);
    cr->Destructor((void*)self, true);
}


/* method/function dispatching -------------------------------------------- */
long cppyy_call_o(cppyy_typehandle_t handle, int method_index,
                  cppyy_object_t self, int numargs, void* args[],
                  cppyy_typehandle_t rettype) {
    void* result = cppyy_allocate(rettype);
    /* TODO: perform call ... */
    return (long)result;
}

static inline G__value cppyy_call_T(cppyy_typehandle_t handle,
        int method_index, cppyy_object_t self, int numargs, void* args[]) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);

    G__InterfaceMethod meth = (G__InterfaceMethod)m->InterfaceMethod();
    G__param libp;
    for (int i = 0; i < numargs; ++i ) {
       G__letint(&libp.para[i], 'u', *(long*)args[i]); // TODO: use actual type code
    }

    G__value result;
    meth(&result, 0, &libp, 0);
    
    return result;
}

void cppyy_call_v(cppyy_typehandle_t handle, int method_index,
                  cppyy_object_t self, int numargs, void* args[]) {
   cppyy_call_T(handle, method_index, self, numargs, args);
}

int cppyy_call_b(cppyy_typehandle_t handle, int method_index,
                 cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return (bool)G__int(result);
}

char cppyy_call_c(cppyy_typehandle_t handle, int method_index,
                  cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return (char)G__int(result);
}

short cppyy_call_h(cppyy_typehandle_t handle, int method_index,
                   cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return (short)G__int(result);
}

long cppyy_call_l(cppyy_typehandle_t handle, int method_index,
                  cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return G__int(result);
}

double cppyy_call_f(cppyy_typehandle_t handle, int method_index,
                    cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return G__double(result);
}

double cppyy_call_d(cppyy_typehandle_t handle, int method_index,
                    cppyy_object_t self, int numargs, void* args[]) {
    G__value result = cppyy_call_T(handle, method_index, self, numargs, args);
    return G__double(result);
}   


cppyy_methptrgetter_t cppyy_get_methptr_getter(cppyy_typehandle_t /*handle*/, int /*method_index*/) {
    return (cppyy_methptrgetter_t)NULL;
}


/* scope reflection information ------------------------------------------- */
int cppyy_is_namespace(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    if (cr.GetClass() && cr->Property())
       return cr->Property() & G__BIT_ISNAMESPACE;
    if (strcmp(cr.GetClassName(), "") == 0)
       return true;
    return false;
}


/* type/class reflection information -------------------------------------- */
char* cppyy_final_name(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    if (cr.GetClass() && cr->Property())
       return type_cppstring_to_cstring(cr->GetName());
    return cppstring_to_cstring(cr.GetClassName());
}

int cppyy_num_bases(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    if (cr.GetClass() && cr->GetListOfBases() != 0)
       return cr->GetListOfBases()->GetSize();
    return 0;
}

char* cppyy_base_name(cppyy_typehandle_t handle, int base_index) {
    TClassRef cr = type_from_handle(handle);
    TBaseClass* b = (TBaseClass*)cr->GetListOfBases()->At(base_index);
    return type_cppstring_to_cstring(b->GetName());
}

int cppyy_is_subtype(cppyy_typehandle_t dh, cppyy_typehandle_t bh) {
    if (dh == bh)
        return 1;
    TClassRef crd = type_from_handle(dh);
    TClassRef crb = type_from_handle(bh);
    return (int)crd->GetBaseClass(crb);
}

size_t cppyy_base_offset(cppyy_typehandle_t dh, cppyy_typehandle_t bh) {
    if (dh == bh)
        return 0;
    TClassRef crd = type_from_handle(dh);
    TClassRef crb = type_from_handle(bh);
    return (size_t)crd->GetBaseClassOffset(crb);
}


/* method/function reflection information --------------------------------- */
int cppyy_num_methods(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    if (cr.GetClass() && cr->GetListOfMethods())
       return cr->GetListOfMethods()->GetSize();
    return 0;
}

char* cppyy_method_name(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    return cppstring_to_cstring(m->GetName());
}

char* cppyy_method_result_type(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    std::string name = TClassEdit::CleanType(m->GetReturnTypeName(), 1);
    return type_cppstring_to_cstring(name);
}

int cppyy_method_num_args(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    return m->GetNargs();
}

int cppyy_method_req_args(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    return m->GetNargs() - m->GetNargsOpt();
}

char* cppyy_method_arg_type(cppyy_typehandle_t handle, int method_index, int arg_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    TMethodArg* arg = (TMethodArg*)m->GetListOfMethodArgs()->At(arg_index);
    return type_cppstring_to_cstring(arg->GetFullTypeName());
}


int cppyy_is_constructor(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    return strcmp(m->GetName(), cr->GetName()) == 0;
}

int cppyy_is_staticmethod(cppyy_typehandle_t handle, int method_index) {
    TClassRef cr = type_from_handle(handle);
    TMethod* m = (TMethod*)cr->GetListOfMethods()->At(method_index);
    return m->Property() & G__BIT_ISSTATIC;
}


/* data member reflection information ------------------------------------- */
int cppyy_num_data_members(cppyy_typehandle_t handle) {
    TClassRef cr = type_from_handle(handle);
    if (cr.GetClass() && cr->GetListOfDataMembers())
       return cr->GetListOfDataMembers()->GetSize();
    return 0;
}

char* cppyy_data_member_name(cppyy_typehandle_t handle, int data_member_index) {
    TClassRef cr = type_from_handle(handle);
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(data_member_index);
    return cppstring_to_cstring(m->GetName());
}

char* cppyy_data_member_type(cppyy_typehandle_t handle, int data_member_index) {
    TClassRef cr = type_from_handle(handle);
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(data_member_index);
    return cppstring_to_cstring(m->GetFullTypeName());
}

size_t cppyy_data_member_offset(cppyy_typehandle_t handle, int data_member_index) {
    TClassRef cr = type_from_handle(handle);
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(data_member_index);
    return m->GetOffset();
}


int cppyy_is_staticdata(cppyy_typehandle_t handle, int data_member_index) {
    TClassRef cr = type_from_handle(handle);
    TDataMember* m = (TDataMember*)cr->GetListOfDataMembers()->At(data_member_index);
    return m->Property() & G__BIT_ISSTATIC;
}


/* misc helper ------------------------------------------------------------ */
void cppyy_free(void* ptr) {
    free(ptr);
}
