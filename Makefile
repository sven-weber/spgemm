SRC_DIR  					= src
INCLUDE_DIR				= include
TARGET   					= build/dphpc
TEST_MACHINES 		= 2
NUMBER_RUNS				= 1
NUMBER_WARMUP			= 0
MATRIX_TARGET			= first
BUILD_DIR         = build

CPP_FILES := $(wildcard $(SRC_DIR)/**/**/*.cpp) $(wildcard $(SRC_DIR)/**/*.cpp) $(wildcard $(SRC_DIR)/*.cpp)
CPP_HEADER_FILES := $(wildcard $(INCLUDE_DIR)/**/**/*.hpp) $(wildcard $(INCLUDE_DIR)/**/*.hpp) $(wildcard $(INCLUDE_DIR)/*.hpp)

.PHONY: debug optimize format

debug: 
	mkdir -p $(BUILD_DIR)
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Debug

optimize: 
	mkdir -p $(BUILD_DIR)
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release

$(BUILD_DIR): debug

run: $(BUILD_DIR)
	@cd $(BUILD_DIR); make
	@echo -e "RUN\t$(TARGET) with $(TEST_MACHINES) machines."
	@./scripts/run $(TEST_MACHINES) ./$(TARGET) "$(MATRIX_TARGET)" $(NUMBER_RUNS) $(NUMBER_WARMUP)

format:
	clang-format -i $(CPP_FILES) $(CPP_HEADER_FILES)

clean:
	rm -rf $(BUILD_DIR)
