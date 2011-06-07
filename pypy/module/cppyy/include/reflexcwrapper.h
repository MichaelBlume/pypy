#ifndef CPPYY_REFLEXCWRAPPER
#define CPPYY_REFLEXCWRAPPER

#ifdef __cplusplus
extern "C" {
#endif // ifdef __cplusplus
    typedef void* cppyy_typehandle_t;
    typedef void* cppyy_object_t;
    typedef void* (*cppyy_methptrgetter_t)(cppyy_object_t);

    /* name to handle */
    cppyy_typehandle_t cppyy_get_typehandle(const char* class_name);
    cppyy_typehandle_t cppyy_get_templatehandle(const char* template_name);

    /* memory management */
    void* cppyy_allocate(cppyy_typehandle_t handle);
    void cppyy_deallocate(cppyy_typehandle_t handle, cppyy_object_t instance);
    void cppyy_destruct(cppyy_typehandle_t handle, cppyy_object_t self);

    /* method/function dispatching */
    void   cppyy_call_v(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    int    cppyy_call_b(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    char   cppyy_call_c(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    short  cppyy_call_h(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    long   cppyy_call_l(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    double cppyy_call_f(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);
    double cppyy_call_d(cppyy_typehandle_t handle, int method_index, cppyy_object_t self, int numargs, void* args[]);

    cppyy_methptrgetter_t cppyy_get_methptr_getter(cppyy_typehandle_t handle, int method_index);

    /* scope reflection information ------------------------------------------- */
    int cppyy_is_namespace(cppyy_typehandle_t handle);

    /* type/class reflection information -------------------------------------- */
    char* cppyy_final_name(cppyy_typehandle_t handle);
    int cppyy_num_bases(cppyy_typehandle_t handle);
    char* cppyy_base_name(cppyy_typehandle_t handle, int base_index);
    int cppyy_is_subtype(cppyy_typehandle_t h1, cppyy_typehandle_t h2);

    /* method/function reflection information */
    int cppyy_num_methods(cppyy_typehandle_t handle);
    char* cppyy_method_name(cppyy_typehandle_t handle, int method_index);
    char* cppyy_method_result_type(cppyy_typehandle_t handle, int method_index);
    int cppyy_method_num_args(cppyy_typehandle_t handle, int method_index);
    char* cppyy_method_arg_type(cppyy_typehandle_t handle, int method_index, int index);

    /* method properties */
    int cppyy_is_constructor(cppyy_typehandle_t handle, int method_index);
    int cppyy_is_staticmethod(cppyy_typehandle_t handle, int method_index);

    /* data member reflection information */
    int cppyy_num_data_members(cppyy_typehandle_t handle);
    char* cppyy_data_member_name(cppyy_typehandle_t handle, int data_member_index);
    char* cppyy_data_member_type(cppyy_typehandle_t handle, int data_member_index);
    size_t cppyy_data_member_offset(cppyy_typehandle_t handle, int data_member_index);

    /* data member properties */
    int cppyy_is_staticdata(cppyy_typehandle_t handle, int data_member_index);

    /* misc helper */
    void cppyy_free(void* ptr);

#ifdef __cplusplus
}
#endif // ifdef __cplusplus

#endif // ifndef CPPYY_REFLEXCWRAPPER
