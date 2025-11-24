project('accelerator_provider', 'c', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'c_std=c11', 'cpp_std=c++14'])

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Options
with_fake = get_option('with_fake')

# Dependencies
thread_dep = dependency('threads')
dl_dep = cc.find_library('dl', required : true)
dcmi_inc = include_directories('/usr/local/dcmi')

# Include directories
inc = include_directories('.')

# Sources
ascend_sources = ['ascend/accelerator.cpp']

# Compiler flags
c_args = []
cpp_args = []
if with_fake
  c_args += '-DASCEND_FAKE_MODE'
  cpp_args += '-DASCEND_FAKE_MODE'
endif

# Shared library
libaccelerator_ascend = shared_library('accelerator_ascend',
  ascend_sources,
  include_directories : [inc, dcmi_inc],
  c_args : c_args,
  cpp_args : cpp_args,
  dependencies : [thread_dep, dl_dep],
  install : true
)

# Tests
gtest_dep = dependency('gtest', main : true, required : false)
if not gtest_dep.found()
  cmake = import('cmake')
  gtest_proj = cmake.subproject('gtest')
  gtest_dep = gtest_proj.dependency('gtest_main')
endif

test_ascend = executable('test_ascend',
  'ascend/tests/test_ascend.cpp',
  include_directories : [inc, dcmi_inc],
  link_with : libaccelerator_ascend,
  dependencies : [gtest_dep, thread_dep, dl_dep],
  c_args : c_args,
  cpp_args : cpp_args
)

test('ascend_provider_test', test_ascend)
