# Cross-platform Makefile for Python project
# Compatible with Windows, Ubuntu, WSL2

# Detect the operating system and shell
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    VENV_DIR := venv
    VENV_ACTIVATE := $(VENV_DIR)\Scripts\activate.bat
    VENV_PYTHON := $(VENV_DIR)\Scripts\python.exe
    VENV_PIP := $(VENV_DIR)\Scripts\pip.exe
    RM := rmdir /s /q
    MKDIR := mkdir
    # Use cmd for Windows commands
    SHELL := cmd.exe
    .SHELLFLAGS := /c
else
    DETECTED_OS := $(shell uname -s)
    PYTHON := python3
    VENV_DIR := venv
    VENV_ACTIVATE := $(VENV_DIR)/bin/activate
    VENV_PYTHON := $(VENV_DIR)/bin/python
    VENV_PIP := $(VENV_DIR)/bin/pip
    RM := rm -rf
    MKDIR := mkdir -p
    SHELL := /bin/bash
    .SHELLFLAGS := -c
endif

# Default target
.PHONY: help
help:
	@echo Available targets:
	@echo   build    - Create virtual environment and install dependencies
	@echo   clean    - Remove virtual environment
	@echo   install  - Install/update dependencies in existing venv
	@echo   info     - Show environment information
	@echo   q1       - Run Part1_Q1/q1.py with activated environment
	@echo   q2       - Run Part1_Q2/q2.py with activated environment

# Create virtual environment and install dependencies
.PHONY: build
build: $(VENV_DIR) install
	@echo Build complete! Virtual environment ready.
	@echo.
	@echo To activate manually:
ifeq ($(OS),Windows_NT)
	@echo   $(VENV_ACTIVATE)
else
	@echo   source $(VENV_ACTIVATE)
endif

# Create virtual environment
$(VENV_DIR):
	@echo Creating virtual environment for $(DETECTED_OS)...
	$(PYTHON) -m venv $(VENV_DIR)
	@echo Virtual environment created at: $(VENV_DIR)

# Install dependencies
.PHONY: install
install: $(VENV_DIR)
	@echo Installing dependencies...
ifeq ($(OS),Windows_NT)
	$(VENV_ACTIVATE) && $(VENV_PYTHON) -m pip install --upgrade pip && $(VENV_PIP) install -r requirements.txt
else
	. $(VENV_ACTIVATE) && $(VENV_PIP) install --upgrade pip && $(VENV_PIP) install -r requirements.txt
endif
	@echo Dependencies installed successfully!

# Clean virtual environment
.PHONY: clean
clean:
	@echo Removing virtual environment...
ifeq ($(wildcard $(VENV_DIR)),)
	@echo Virtual environment does not exist.
else
	$(RM) $(VENV_DIR)
	@echo Virtual environment removed.
endif

# Show environment information
.PHONY: info
info:
	@echo Environment Information:
	@echo   OS: $(DETECTED_OS)
	@echo   Python: $(PYTHON)
	@echo   Virtual Environment: $(VENV_DIR)
	@echo   Activation Script: $(VENV_ACTIVATE)
	@echo.
	@echo Project Structure:
	@echo   requirements.txt - Python dependencies
	@echo   Makefile - Build automation
	@echo   $(VENV_DIR)/ - Virtual environment (created by 'make build')

# Rebuild - clean and build
.PHONY: rebuild
rebuild: clean build
	@echo Rebuild complete!

# Run Q1 script
.PHONY: q1
q1: $(VENV_DIR)
	@echo Running Q1...
ifeq ($(OS),Windows_NT)
	$(VENV_ACTIVATE) && cd Part1_Q1 && ..\$(VENV_PYTHON) q1_local.py && cd ..
else
	. $(VENV_ACTIVATE) && cd Part1_Q1 && ../$(VENV_PYTHON) q1_local.py && cd ..
endif

# Run Q2 script
.PHONY: q2
q2: $(VENV_DIR)
	@echo Running Q2...
ifeq ($(OS),Windows_NT)
	$(VENV_ACTIVATE) && cd Part1_Q2 && ..\$(VENV_PYTHON) q2_local.py && cd ..
else
	. $(VENV_ACTIVATE) && cd Part1_Q2 && ../$(VENV_PYTHON) q2_local.py && cd ..
endif