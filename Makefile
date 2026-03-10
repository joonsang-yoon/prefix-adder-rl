SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

.SUFFIXES:
.DELETE_ON_ERROR:

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

GEN_DIR ?= $(MAKEFILE_DIR)generated
VERILOG_OUTPUT_DIR ?= $(GEN_DIR)/verilog
SEARCH_OUTPUT_DIR ?= $(GEN_DIR)/search
EVAL_OUTPUT_DIR ?= $(GEN_DIR)/eval

PROJECT ?= TopLevelModule
MODULE ?= TopLevelModule.ExamplePrefixAdder
ALGORITHM ?= tabular
BACKEND ?= librelane
WIDTH ?= 8
EPISODES ?= 16
SMOKE_EPISODES ?= 4
SEED ?= 1
LEARNING_RATE ?=
TEMPERATURE ?= 1.0
HIDDEN_SIZE ?= 32
GRADIENT_CLIP ?= 5.0
BASELINE_MOMENTUM ?= 0.9
CHECKPOINT_INTERVAL ?= 0
POLICY_INIT ?=
RUN_LABEL ?=
CLEAN_OUTPUT ?= true
SKIP_WARM_STARTS ?= false
DAG_SUMMARY_MODE ?= usage-weighted
ACTION_CONTEXT_MODE ?= self-attention
TOPOLOGY ?= $(MAKEFILE_DIR)example_topologies/dependent_balanced_8.json
REGISTER_OUTPUTS ?= true
CLOCK_PERIOD ?= 5.0
LIBRELANE_CMD ?= librelane

MILL ?= $(MAKEFILE_DIR)mill
MILL_JOBS ?= $(shell n="$$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"; [ "$$n" -gt 4 ] && echo 4 || echo "$$n")
MILL_OPTS ?= -i -j $(MILL_JOBS)
MILL_CMD := bash "$(MILL)" $(MILL_OPTS)

FORMAT_MODULES_QUERY = {TopLevelModule,ExternalModule,PrefixAdderLib,PrefixRLCore,PrefixTabularRL,PrefixDeepRL,PrefixUtils}

.DEFAULT_GOAL := help

empty :=
space := $(empty) $(empty)
comma := ,
lparen := (
rparen := )

define module_to_path
$(strip $(subst $(space),,$(subst $(comma),_,$(subst $(rparen),,$(subst $(lparen),_,$(subst .,/,$(1)))))))
endef

MODULE_PATH := $(call module_to_path,$(MODULE))
TARGET_DIR ?= $(VERILOG_OUTPUT_DIR)/$(MODULE_PATH)

.PHONY: help verilog macro evaluate search search-tabular search-deep smoke-evaluate smoke-search-tabular smoke-search-deep sweep-search test reformat check-format check clean distclean elaborate-help

help: ## Show this help
	@echo "Usage:"
	@echo "  make <target> [VARIABLE=value]"
	@echo ""
	@echo "Common variables:"
	@echo "  MODULE=$(MODULE)"
	@echo "  TARGET_DIR=$(TARGET_DIR)"
	@echo "  ALGORITHM=$(ALGORITHM)"
	@echo "  BACKEND=$(BACKEND)"
	@echo "  WIDTH=$(WIDTH)"
	@echo "  TOPOLOGY=$(TOPOLOGY)"
	@echo "  EPISODES=$(EPISODES)"
	@echo "  SMOKE_EPISODES=$(SMOKE_EPISODES)"
	@echo "  SEED=$(SEED)"
	@echo "  LEARNING_RATE=$(LEARNING_RATE)"
	@echo "  TEMPERATURE=$(TEMPERATURE)"
	@echo "  HIDDEN_SIZE=$(HIDDEN_SIZE)"
	@echo "  GRADIENT_CLIP=$(GRADIENT_CLIP)"
	@echo "  BASELINE_MOMENTUM=$(BASELINE_MOMENTUM)"
	@echo "  CHECKPOINT_INTERVAL=$(CHECKPOINT_INTERVAL)"
	@echo "  POLICY_INIT=$(POLICY_INIT)"
	@echo "  RUN_LABEL=$(RUN_LABEL)"
	@echo "  CLEAN_OUTPUT=$(CLEAN_OUTPUT)"
	@echo "  SKIP_WARM_STARTS=$(SKIP_WARM_STARTS)"
	@echo "  DAG_SUMMARY_MODE=$(DAG_SUMMARY_MODE)"
	@echo "  ACTION_CONTEXT_MODE=$(ACTION_CONTEXT_MODE)"
	@echo "  LIBRELANE_CMD=$(LIBRELANE_CMD)"
	@echo "  CLOCK_PERIOD=$(CLOCK_PERIOD)"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ { t[++n] = $$1; d[n] = $$2; if (length($$1) > m) m = length($$1) } END { for (i=1; i<=n; i++) printf "  %-" m "s %s\n", t[i], d[i] }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Examples:"
	@echo "  make verilog MODULE=TopLevelModule.ExamplePrefixAdder"
	@echo "  make macro WIDTH=8 TOPOLOGY=example_topologies/dependent_ripple_8.json"
	@echo "  make evaluate WIDTH=8 TOPOLOGY=example_topologies/dependent_balanced_8.json BACKEND=synthetic"
	@echo "  make search ALGORITHM=tabular BACKEND=synthetic WIDTH=8 EPISODES=32 RUN_LABEL=seed1"
	@echo "  make search ALGORITHM=deep BACKEND=librelane WIDTH=8 EPISODES=32 HIDDEN_SIZE=48 DAG_SUMMARY_MODE=attention-weighted ACTION_CONTEXT_MODE=mean-residual CHECKPOINT_INTERVAL=8 RUN_LABEL=abl_b"
	@echo "  make search-deep BACKEND=synthetic WIDTH=6 EPISODES=12 POLICY_INIT=generated/search/deep/seed1/logs/policy/policy_final.json SKIP_WARM_STARTS=true RUN_LABEL=finetune"

verilog: ## Generate SystemVerilog for MODULE using the reflection-based Elaborate wrapper
	@mkdir -p "$(TARGET_DIR)"
	$(MILL_CMD) $(PROJECT).runMain Elaborate "$(MODULE)" --target-dir "$(TARGET_DIR)"
	@echo "Generated SystemVerilog in: $(TARGET_DIR)"

macro: ## Generate PrefixAdderMacro RTL from WIDTH/TOPOLOGY using scripts/elaborate_prefix_adder.sh
	bash "$(MAKEFILE_DIR)scripts/elaborate_prefix_adder.sh" \
	  --width "$(WIDTH)" \
	  --topology "$(TOPOLOGY)" \
	  --register-outputs "$(REGISTER_OUTPUTS)" \
	  --target-dir "$(VERILOG_OUTPUT_DIR)/PrefixAdderMacro_$(WIDTH)"

evaluate: ## Evaluate one dependent-tree topology with BACKEND=librelane or BACKEND=synthetic
	WIDTH="$(WIDTH)" TOPOLOGY="$(TOPOLOGY)" OUTPUT_ROOT="$(EVAL_OUTPUT_DIR)" BACKEND="$(BACKEND)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)" LIBRELANE_CMD="$(LIBRELANE_CMD)" \
	  bash "$(MAKEFILE_DIR)scripts/evaluate_topology.sh"

search: ## Run end-to-end RL topology search with ALGORITHM=tabular|deep and BACKEND=librelane|synthetic
	ALGORITHM="$(ALGORITHM)" WIDTH="$(WIDTH)" EPISODES="$(EPISODES)" SEED="$(SEED)" LEARNING_RATE="$(LEARNING_RATE)" TEMPERATURE="$(TEMPERATURE)" HIDDEN_SIZE="$(HIDDEN_SIZE)" GRADIENT_CLIP="$(GRADIENT_CLIP)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" POLICY_INIT="$(POLICY_INIT)" RUN_LABEL="$(RUN_LABEL)" CLEAN_OUTPUT="$(CLEAN_OUTPUT)" SKIP_WARM_STARTS="$(SKIP_WARM_STARTS)" DAG_SUMMARY_MODE="$(DAG_SUMMARY_MODE)" ACTION_CONTEXT_MODE="$(ACTION_CONTEXT_MODE)" OUTPUT_ROOT="$(SEARCH_OUTPUT_DIR)/$(ALGORITHM)" BACKEND="$(BACKEND)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)" LIBRELANE_CMD="$(LIBRELANE_CMD)" \
	  bash "$(MAKEFILE_DIR)scripts/run_search.sh"

search-tabular: ## Run the tabular RL search loop
	@$(MAKE) search ALGORITHM=tabular BACKEND="$(BACKEND)" WIDTH="$(WIDTH)" EPISODES="$(EPISODES)" SEED="$(SEED)" LEARNING_RATE="$(LEARNING_RATE)" TEMPERATURE="$(TEMPERATURE)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" POLICY_INIT="$(POLICY_INIT)" RUN_LABEL="$(RUN_LABEL)" CLEAN_OUTPUT="$(CLEAN_OUTPUT)" SKIP_WARM_STARTS="$(SKIP_WARM_STARTS)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)" LIBRELANE_CMD="$(LIBRELANE_CMD)"

search-deep: ## Run the deep RL search loop
	@$(MAKE) search ALGORITHM=deep BACKEND="$(BACKEND)" WIDTH="$(WIDTH)" EPISODES="$(EPISODES)" SEED="$(SEED)" LEARNING_RATE="$(LEARNING_RATE)" TEMPERATURE="$(TEMPERATURE)" HIDDEN_SIZE="$(HIDDEN_SIZE)" GRADIENT_CLIP="$(GRADIENT_CLIP)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" POLICY_INIT="$(POLICY_INIT)" RUN_LABEL="$(RUN_LABEL)" CLEAN_OUTPUT="$(CLEAN_OUTPUT)" SKIP_WARM_STARTS="$(SKIP_WARM_STARTS)" DAG_SUMMARY_MODE="$(DAG_SUMMARY_MODE)" ACTION_CONTEXT_MODE="$(ACTION_CONTEXT_MODE)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)" LIBRELANE_CMD="$(LIBRELANE_CMD)"

smoke-evaluate: ## Run a one-off synthetic evaluation without requiring LibreLane
	@$(MAKE) evaluate BACKEND=synthetic WIDTH="$(WIDTH)" TOPOLOGY="$(TOPOLOGY)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)"

smoke-search-tabular: ## Run a short tabular RL smoke test with the synthetic backend
	@$(MAKE) search-tabular BACKEND=synthetic WIDTH="$(WIDTH)" EPISODES="$(SMOKE_EPISODES)" SEED="$(SEED)" LEARNING_RATE="$(LEARNING_RATE)" TEMPERATURE="$(TEMPERATURE)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)"

smoke-search-deep: ## Run a short deep RL smoke test with the synthetic backend
	@$(MAKE) search-deep BACKEND=synthetic WIDTH="$(WIDTH)" EPISODES="$(SMOKE_EPISODES)" SEED="$(SEED)" LEARNING_RATE="$(LEARNING_RATE)" TEMPERATURE="$(TEMPERATURE)" HIDDEN_SIZE="$(HIDDEN_SIZE)" GRADIENT_CLIP="$(GRADIENT_CLIP)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" DAG_SUMMARY_MODE="$(DAG_SUMMARY_MODE)" ACTION_CONTEXT_MODE="$(ACTION_CONTEXT_MODE)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)"

sweep-search: ## Run a matrix sweep using scripts/sweep_search.sh (configure via *_LIST variables)
	ALGORITHM="$(ALGORITHM)" BACKEND="$(BACKEND)" WIDTH="$(WIDTH)" EPISODES="$(EPISODES)" SMOKE_EPISODES="$(SMOKE_EPISODES)" OUTPUT_ROOT_BASE="$(SEARCH_OUTPUT_DIR)/$(ALGORITHM)" RUN_PREFIX="$(RUN_PREFIX)" SEEDS_LIST="$(SEEDS_LIST)" TEMPERATURES_LIST="$(TEMPERATURES_LIST)" LEARNING_RATES_LIST="$(LEARNING_RATES_LIST)" HIDDEN_SIZES_LIST="$(HIDDEN_SIZES_LIST)" DAG_SUMMARY_MODES_LIST="$(DAG_SUMMARY_MODES_LIST)" ACTION_CONTEXT_MODES_LIST="$(ACTION_CONTEXT_MODES_LIST)" BASELINE_MOMENTUM="$(BASELINE_MOMENTUM)" CHECKPOINT_INTERVAL="$(CHECKPOINT_INTERVAL)" REGISTER_OUTPUTS="$(REGISTER_OUTPUTS)" CLOCK_PERIOD="$(CLOCK_PERIOD)" LIBRELANE_CMD="$(LIBRELANE_CMD)" MILL_JOBS="$(MILL_JOBS)" 	  bash "$(MAKEFILE_DIR)scripts/sweep_search.sh"

test: ## Run Scala tests for utilities, topology JSON, Pareto-frontier behavior, both RL policies, and the synthetic backend
	$(MILL_CMD) "$(FORMAT_MODULES_QUERY).test"

reformat: ## Reformat Scala sources (scalafmt via Mill)
	$(MILL_CMD) "$(FORMAT_MODULES_QUERY).reformat"

check-format: ## Check Scala formatting (CI-friendly)
	$(MILL_CMD) "$(FORMAT_MODULES_QUERY).checkFormat"

check: ## Run formatting checks, unit tests, and synthetic end-to-end smoke tests
	@$(MAKE) check-format
	@$(MAKE) test
	@$(MAKE) smoke-evaluate
	@$(MAKE) smoke-search-tabular WIDTH=4 SMOKE_EPISODES=2
	@$(MAKE) smoke-search-deep WIDTH=4 SMOKE_EPISODES=2 HIDDEN_SIZE=16

clean: ## Remove generated Verilog files and search/evaluation artifacts
	rm -rf "$(GEN_DIR)"

distclean: clean ## Remove generated artifacts plus Mill out/
	rm -rf "$(MAKEFILE_DIR)out"

elaborate-help: ## Show full Elaborate/ChiselStage options
	@$(MILL_CMD) $(PROJECT).runMain Elaborate --help
