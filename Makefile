CXX = g++-10 -fopenmp

# CPPFLAGS = -std=c++17 -g
CPPFLAGS = -std=c++17 -O3
#-Wall: open all warnings

INCLUDES = -I include/ -I /usr/local/Cellar/eigen/3.3.7/include/eigen3 
#INCLUDES = -I include/ -I Eigen_3_3/

## gtest library as static library, with library file *.a stored in ./lib/
GTEST_LIBFLAGS = ./lib/libgtest.a
## gtest library as shared library, with library file *.so stored in /usr/local/lib/; require sudo; some configuration might be needed
#GTEST_LIBFLAGS = -lgtest


DYLBFLAGS = -dynamiclib
#DYLBFLAGS = -shared

CURRENT_VERSION = 1.0.1
CURRENT_SONAME = 1






DEPFLAGS = -MT $@ -MMD -MP -MF $(DEP_PATH)/$*.d
RM = rm

SRC_PATH = source
BUILD_PATH = build
BIN_PATH = bin
DEP_PATH = deps
TEST_PATH = test
LIB_PATH = lib
EXAMPLE_PATH = example

LIB_FILE_NAME = $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION)

SRC_EXT = cpp

SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
DEPFILES = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(DEP_PATH)/%.d)

TESTSOURCES = $(shell find $(TEST_PATH) -name 'test_*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
TESTDEPFILES = $(TESTSOURCES:$(TEST_PATH)/%.$(SRC_EXT)=$(DEP_PATH)/%.d)
TESTOBJECTS = $(TESTSOURCES:$(TEST_PATH)/%.$(SRC_EXT)=$(TEST_PATH)/obj/%.o)
TESTEXEC = $(TESTSOURCES:$(TEST_PATH)/%.$(SRC_EXT)=$(BIN_PATH)/%)

EXAMPLESOURCES = $(shell find $(EXAMPLE_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
EXAMPLEEXEC = $(EXAMPLESOURCES:$(EXAMPLE_PATH)/%.$(SRC_EXT)=$(BIN_PATH)/%)
EXAMPLEEXEC_NAME = $(EXAMPLEEXEC:$(BIN_PATH)/%=%)








### Make all, default command

.PHONY: all
all: $(EXAMPLEEXEC) | $(LIB_FILE_NAME)
	@for filename in $(EXAMPLEEXEC_NAME) ; do \
		echo "Make symlink: $$filename --> $(BIN_PATH)/$$filename" ; \
		$(RM) -f $$filename ; \
		ln -s $(BIN_PATH)/$$filename $$filename ; \
	done
	
	













### Create directories

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(BUILD_PATH)
	@mkdir -p $(BIN_PATH)
	@mkdir -p $(DEP_PATH)
	@mkdir -p $(LIB_PATH)
	@mkdir -p $(TEST_PATH)/obj










### Create Shared Library

.PHONY: $(BUILD_PATH)/%.o
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp | dirs
	@echo "Compile $< to output object $@ for shared library, based on auto-dependency in $(DEP_PATH)/$*.d"
	@$(CXX) $(DEPFLAGS) $(CPPFLAGS) $(INCLUDES) -fPIC -c $< -o $@


$(LIB_FILE_NAME): $(OBJECTS)
	@echo "Output shared library $@, current version is $(CURRENT_VERSION)"
	@echo "Warning: Depending on the environment, you should use either option '-shared' in linux machine, or '-dynamiclib' in unix/OS machine"
	@$(CXX) $(DYLBFLAGS) -o $@ $(OBJECTS)
	@echo "\n--------------------------------------------------------------------------------------------------"
	@echo "Shared library $(LIB_FILE_NAME) is successfully created."
	@echo "--------------------------------------------------------------------------------------------------\n\n"

sharedlib: $(LIB_FILE_NAME)


.PHONY: doxygen
doxygen:
	@echo "Create Documents for the shared library of sgdg, in doc/html, by doxygen"
	@doxygen



#sharedlib_linux: dirs
#sharedlib_linux: $(OBJECTS)
#sharedlib_linux:
#	@echo "Output shared library $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION), current version is $(CURRENT_VERSION) with soname $(CURRENT_SONAME)"
#	@$(CXX) -shared -Wl,-soname,libsgdg.so.$(CURRENT_SONAME) -o $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION) $(OBJECTS)
#	@echo "Configure dynamic linker run-time bindings by ldconfig"
#	@ldconfig -n $(LIB_PATH)



### Old version of sharedlib in linux machine
###
### @$(CXX) -shared -Wl,-soname,libsgdg.so.$(CURRENT_SONAME) -o $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION) $(OBJECTS)
### @ldconfig -n $(LIB_PATH)
###
### in unix/OS -shared should be replaced by -dynamiclib, and -soname should be replaced by -install_name
### in unix/OS we do not have ldconfig; so we need some command such as ld or dyld to "Configure dynamic linker run-time bindings", e.g.
### 	binding between libsgdg.so.1 (only some sort of reference) and libsgdg.so.1.0.1 (real library name)
###
###
### There is no need to export this path, if we use e.g. libsgdg.so.1.0.1 directly and we do not use -Wl,-soname (linux) or -Wl,-install_name (OS,unix) option
###
### export LD_LIBRARY_PATH=lib
### export the library path as environment variable 
### check the path is well-defined by
###		echo $LD_LIBRARY_PATH









### Create object and exectuable for examples

.PHONY: $(BIN_PATH)/%.o | dirs
$(BIN_PATH)/%.o : $(EXAMPLE_PATH)/%.$(SRC_EXT)
	@echo "Compile $< to output object $@ for examples, based on auto-dependency in $(DEP_PATH)/$*.d"
	@$(CXX) $(DEPFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

.PRECIOUS: $(BIN_PATH)/%.o
.PHONY: $(BIN_PATH)/%
$(BIN_PATH)/% : $(BIN_PATH)/%.o | sharedlib
	@echo "Link all the shared library of sgdg and $< to get executable $@ for examples\n"
	@$(CXX) -o $@ -L $(LIB_PATH)/ $< $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION)














### Create test objects and exectuables, and run the tests

.PHONY: $(TEST_PATH)/obj/%.o
$(TEST_PATH)/obj/%.o: $(TEST_PATH)/%.$(SRC_EXT) | dirs
	@echo "Compile test source $< to output object $@ for tests, based on auto-dependency in $(DEP_PATH)/$*.d"
	@$(CXX) $(DEPFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@ 

.PHONY: $(TEST_PATH)/main
$(TEST_PATH)/main: $(TESTOBJECTS) | $(LIB_FILE_NAME)
	@echo "Link all the shared library of sgdg and $^ to get executable $@ for tests\n"
	@$(CXX) -o $@ $(TESTOBJECTS) $(LIB_FILE_NAME) $(GTEST_LIBFLAGS) -lpthread

.PHONY: alltest
alltest: $(TEST_PATH)/main
	@echo "Run all tests"
	@./$<






### Installation of gtest as static library, in HPCC, with no access to sudo and /usr/local/lib/ ###
#  git clone https://github.com/google/googletest.git
#  cd googletest/
#  git checkout v1.8.x   # move to stable version
#  module load CMake     # if not loaded yet
#  mkdir build
#  cd build
#  cmake ..
#  cp -r ./googletest/include/gtest/ ../sparse-grid-DG/include/    # copy gtest header to your local include path
#  cp lib*.a ../../../../sparse-grid-DG/lib/                       # copy gtest static library to your local library path; file names are libgtest.a and libgtest_main.a


### Note of gtest in HPCC ###
### 1. error of "/opt/software/binutils/2.30-GCCcore-7.3.0/bin/ld: cannot find -lgtest", when linking the test objects and libraries
###       shared library is not found; you could use static library. (to use shared library you need to change some option in CMakeList.txt when you make gtest library; default version in the cmake set-up is static lib); in such case you can not use -L./lib/libgtest.a, otherwise there are multiple errors of undefined references. Shared library .a file is treated like object .o files.
### 2. multiple errors when running ./test/main, like "./test/main: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by ./test/main)" or  "./test/main: /lib64/libstdc++.so.6: version `CXXABI_1.3.9' not found (required by ./test/main)"
###        GLIBCXX or CXXABI should be part of GCC library; choose the proper version of GCC and this error goes away. This might be fixed by a "module load GCC" (GCC/5 or higher are ok). See https://askubuntu.com/questions/575505/glibcxx-3-4-20-not-found-how-to-fix-this-error. This reload of GCC does not affect the make all command.
###        Update: it seems numpy requires icc or ifort and GCCCore-4.9.2, which contradicts with GCC, so that the batch.py is not working; this is not a problem in ubuntu













### Clean everything, EXCLUDING the old version of libraries

.PHONY: clean
clean:
	@for filename in $(EXAMPLEEXEC_NAME) ; do \
		echo "Deleting the symlinks : $$filename" ; \
		$(RM) -f $$filename ; \
	done
	@echo "Deleting directories, of executable in $(BIN_PATH), dependency in $(DEP_PATH), objects in $(BUILD_PATH), and documents in doc/html"
	@echo "Warning: We would NOT delete $(LIB_PATH) and its OLD library, but we delete the current version $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION)"
	@$(RM) -f -r $(BUILD_PATH)
	@$(RM) -f -r $(DEP_PATH)
	@$(RM) -f -r $(BIN_PATH)
	@$(RM) -f $(LIB_PATH)/libsgdg.so.$(CURRENT_VERSION)
	@$(RM) -f -r doc/html
	@echo "Deleting test objects and exectuable"
	@$(RM) -f -r $(TEST_PATH)/obj
	@$(RM) -f $(TEST_PATH)/main







$(DEPDIR): ;


$(DEPFILES):
include $(wildcard $(DEPFILES))
