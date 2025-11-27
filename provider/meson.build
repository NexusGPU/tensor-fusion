project('accelerator_provider', 'c', 'cpp',
  version : '0.1.0',
  default_options : [
    'warning_level=3',
    'c_std=c11',
    'cpp_std=c++17',
    'buildtype=release',
    'b_ndebug=if-release'
  ]
)

# Get compilers
cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Build options
build_stub = get_option('build_stub')
build_ascend = get_option('build_ascend')
build_tests = get_option('build_tests')

# Common dependencies
thread_dep = dependency('threads')
dl_dep = cc.find_library('dl', required : true)

# Include directories
inc = include_directories('.')

# ============================================================================
# Stub implementation
# ============================================================================
if build_stub
  stub_sources = files('stub/accelerator.c')

  libaccelerator_stub = shared_library('accelerator_stub',
    stub_sources,
    include_directories : inc,
    dependencies : [thread_dep],
    install : true,
    install_dir : get_option('libdir') / 'tensor-fusion'
  )

  # Stub test
  if build_tests
    test_accelerator = executable('test_accelerator',
      'test/test_accelerator.c',
      include_directories : inc,
      link_with : libaccelerator_stub,
      dependencies : [thread_dep],
      build_by_default : true
    )

    test('stub_basic_test', test_accelerator)
  endif
endif

# ============================================================================
# Ascend implementation
# ============================================================================
if build_ascend
  # DCMI include directory (configurable)
  dcmi_include = get_option('dcmi_include_dir')
  if dcmi_include == ''
    dcmi_include = '/usr/local/dcmi'
  endif
  dcmi_inc = include_directories(dcmi_include, is_system : true)

  ascend_sources = files('ascend/accelerator.cpp')

  ascend_cpp_args = ['-pthread']

  libaccelerator_ascend = shared_library('accelerator_ascend',
    ascend_sources,
    include_directories : [inc, dcmi_inc],
    cpp_args : ascend_cpp_args,
    dependencies : [thread_dep, dl_dep],
    install : true,
    install_dir : get_option('libdir') / 'tensor-fusion'
  )

  # Ascend tests
  if build_tests
    # Comprehensive test (recommended)
    test_ascend_comprehensive = executable('test_ascend_comprehensive',
      'ascend/tests/test_ascend_comprehensive.cpp',
      include_directories : [inc, dcmi_inc],
      link_with : libaccelerator_ascend,
      cpp_args : ascend_cpp_args,
      dependencies : [thread_dep, dl_dep],
      build_by_default : true,
      install : false
    )

    # Note: Ascend tests require hardware and root privileges
    # Run manually with: sudo -E ./build/test_ascend_comprehensive
    message('Ascend tests built. Run manually with root privileges.')
    message('Example: sudo -E DCMI_LIB_PATH=/usr/local/dcmi/libdcmi.so ./build/test_ascend_comprehensive')

    # Legacy fake partition test
    test_ascend_fake = executable('test_ascend_fake',
      'test/ascend_partition_fake.c',
      include_directories : [inc, dcmi_inc],
      link_with : libaccelerator_ascend,
      dependencies : [thread_dep, dl_dep],
      build_by_default : false,
      install : false
    )
  endif
endif

# ============================================================================
# Install headers
# ============================================================================
install_headers(
  'accelerator.h',
  'limiter.h',
  subdir : 'tensor-fusion'
)

# ============================================================================
# Summary
# ============================================================================
summary({
  'Stub implementation': build_stub,
  'Ascend implementation': build_ascend,
  'Build tests': build_tests,
  'DCMI include dir': dcmi_include,
  'Install prefix': get_option('prefix'),
  'Library dir': get_option('libdir') / 'tensor-fusion',
}, section: 'Configuration')
