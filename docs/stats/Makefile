SCRIPTPATH:=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECTPATH:=$(realpath $(SCRIPTPATH)/../..)

STATSMD = $(PROJECTPATH)/docs/stats.md
STATSPY = $(PROJECTPATH)/docs/stats/stats.py
UNITTEST_GLOB = $(PROJECTPATH)/test/Target/SubstraitPB/Import/*.textpb

.DELETE_ON_ERROR:

all: $(STATSMD)

$(STATSMD): $(wildcard $(UNITTEST_GLOB)) $(STATSPY)
	python $(STATSPY) --messages "$(UNITTEST_GLOB)" > $@
