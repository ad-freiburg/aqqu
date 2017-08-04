STANFORD_CORENLP_DIR=lib/stanford-corenlp-full-2014-08-27
PARSER_PORT=12345
SPARQL_ENDPOINT=http://localhost:8999/sparql/
HAPROXY_CFG=haproxy/haproxy.cfg
HAPROXY_BIN=haproxy/haproxy-1.5.9/haproxy
VARNISH_CFG=haproxy/varnish.vcl
VARNISH_HACFG=haproxy/varnish_virtuoso.vcl
VARNISH_BIND=metropolis:9000
VARNISH_HABIND=metropolis:9001
VARNISH_BIN=/home/haussmae/temp/varnish/sbin/varnishd
VARNISH_LIB_DIR=/home/haussmae/temp/varnish/lib/varnish
DATA = data
LOCAL_DATA = local-data
PYTHONPATH = $(shell pwd):$(shell pwd)/venv/lib/python2.7/site-packages/
PYTHON = PYTHONPATH=$(PYTHONPATH) python
PYPY = PYTHONPATH=$(PYTHONPATH) ~/pypy/pypy/bin/pypy
MEDIATOR_INDEX_PREFIX = $(DATA)/mediator_index/mediator_index_fast
MEDIATOR_FACTS = $(LOCAL_DATA)/mediator_facts_clean.txt
VIRTUOSO_BINARY = data/virtuoso/install/bin/virtuoso-t
WEBSERVER_CONFIG = old_config.cfg
PUBLISH_DATA = publish_data

NAME_MID_MAP = $(DATA)/name-mapping

# PRE-PROCESSING DATA

# Remove sentences < 2 entities and filter out garbage sentences.
$(DATA)/entitysentences.txt: $(NAME_MID_MAP) $(DATA)/entitysentences_unfiltered.txt
	$(PYTHON) scripts/filter_entitysentences.py $^ > $@

# Parse all sentences.
$(DATA)/entitysentences_parsed.txt: $(DATA)/entitysentences.txt
	$(PYTHON) scripts/add_parses.py --njobs 50 $^ $@

# Parse all sentences with generic target.
%_parsed.txt: %
	$(PYTHON) scripts/add_parses.py --njobs 50 $^ $@

# Find binary relations that match the entities.
$(DATA)/entitysentences_binary_relations.txt: $(DATA)/entitysentences_parsed.txt
	$(PYTHON) scripts/relation-contexts/find_binary_relations.py --njobs 16 $^ $@

# Find binary relations for generic input file.
%_binary_relations.txt: %
	$(PYPY) scripts/relation-contexts/find_binary_relations.py --njobs 16 $^ $@

# Find mediated relations that match the entities.
$(DATA)/entitysentences_mediated_relations.txt: $(DATA)/entitysentences_parsed.txt
	$(PYTHON) scripts/relation-contexts/find_mediated_relations.py --njobs 8 $^ $@ $(MEDIATOR_INDEX_PREFIX) $(MEDIATOR_FACTS)

# Find mediated relations that match the entities.
%_mediated_relations.txt: %
	$(PYTHON) scripts/relation-contexts/find_mediated_relations.py --njobs 6 $^ $@ $(MEDIATOR_INDEX_PREFIX) $(MEDIATOR_FACTS)

# Extract mediated facts from Freebase.
$(DATA)/mediator_facts.txt:
	$(PYTHON) scripts/freebase-data/extract_mediator_facts.py > $@

# Extract reverse relations from Freebase.
$(DATA)/reverse-relations:
	$(PYTHON) scripts/freebase-data/extract_reverse_relations.py > $@

# Extract reverse relations from Freebase.
$(DATA)/relation-lemmas:
	$(PYTHON) scripts/freebase-data/extract_relation_lemmas.py > $@

# Extract mediator relations from Freebase.
$(DATA)/mediator-relations: $(DATA)/type-ids
	$(PYTHON) scripts/freebase-data/extract_mediator_relations.py $^ > $@

# Extract relation target types from Freebase.
$(DATA)/relation-expected-types:
	$(PYTHON) scripts/freebase-data/extract_relation_expected_types.py > $@

# Extract relation words from Freebase.
$(DATA)/relation-words: data/wikipedia_sentences/entitysentences_binary_relations.txt
	$(PYTHON) scripts/relation-contexts/extract_relation_words.py $^ > $@

# Extract relation words from Freebase.
$(DATA)/mediated-relation-words_: data/wikipedia_sentences/entitysentences_mediated_relations.txt
	$(PYTHON) scripts/relation-contexts/extract_relation_words.py $^ > $@

# Extract relation counts from Freebase.
$(DATA)/relation-counts:
	$(PYTHON) scripts/freebase-data/extract_relation_counts.py > $@

# Extract relation target types from Freebase.
$(DATA)/relation-target-type-distributions:
	$(PYTHON) scripts/freebase-data/extract_relation_target_type_distribution.py > $@

# Extract entity aliases from Freebase.
$(DATA)/entity-aliases:
	$(PYTHON) scripts/freebase-data/extract_entity_aliases.py > $@

# Extract entity names from Freebase.
$(DATA)/entity-names:
	$(PYTHON) scripts/freebase-data/extract_entity_names.py > $@

# Extract entity wiki names from Freebase.
$(DATA)/entity-wiki-names:
	$(PYTHON) scripts/freebase-data/extract_entity_wiki_names.py > $@

# Extract replacement mids from Freebase.
$(DATA)/replacement-mids:
	$(PYTHON) scripts/freebase-data/extract_replacement_mids.py > $@

# Extract type idsfrom Freebase.
$(DATA)/type-ids:
	$(PYTHON) scripts/freebase-data/extract_type_id_map.py > $@

# Create entity list.
$(DATA)/entity-list: $(DATA)/replacement-mids $(DATA)/name-mapping $(DATA)/entity-scores $(DATA)/entity-names $(DATA)/entity-aliases $(DATA)/entity-wiki-names
	$(PYPY) scripts/entity-list/create_entity_list.py $^ > $@

# Create crosswikis alias list.
$(DATA)/entity-crosswikis-aliases: $(DATA)/entity-wiki-names $(DATA)/crosswikis/lnrm.dict_filtered
	$(PYPY) scripts/entity-list/create_crosswikis_aliases.py $^ > $@

# Create entity surface map.
$(DATA)/entity-surface-map: $(DATA)/entity-list $(DATA)/entity-crosswikis-aliases $(DATA)/tokenizer.abbreviations
	$(PYPY) scripts/entity-list/create_entity_surface_map.py $^ > $@

# Clean mediated facts.
$(DATA)/mediator_facts_clean.txt: $(DATA)/mediator_facts.txt
	grep -vE "type.object.type|http://www.w3.org/2000/01/rdf-schema#label|http://www.w3.org/1999/02/22-rdf-syntax-ns#type|common.topic.description|award.award_nomination.notes_description|common.webpage.description|award.award_honor.notes_description" $^> $@

# Here for documentation reasons.
$(DATA)/entity-types: $(DATA)/entity-list $(DATA)/freebase/type.object.type.gz
	$(PYPY) scripts/freebase-data/extract_entity_types.py data/entity-list  data/freebase/type.object.type.gz > $@

# Here for documentation reasons.
$(DATA)/word-entity-type-counts: $(DATA)/entity-scores $(DATA)/entity-types
	$(PYPY) scripts/relation-contexts/extract_word_type_counts.py --entity_scores data/entity-scores --parallel data/entity-types data/clueweb_sentences/annotationsClueWeb12_01.tgz_sentences.gz_parsed.gz  data/clueweb_sentences/annotationsClueWeb12_00.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_02.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_03.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_04.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_05.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_06.tgz_sentences.gz_parsed.gz data/clueweb_sentences/annotationsClueWeb12_07.tgz_sentences.gz_parsed.gz > $@

# Add mediator type hints. Updates via SPARQL must be allowed!
fix-freebase:
	$(PYTHON) scripts/fix_freebase.py

start-parser:
	cd corenlp-frontend; ant run

start-webserver:
	$(PYTHON) webserver/translation_webserver.py --config-file $(WEBSERVER_CONFIG)

start-virtuoso:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db/virtuoso.ini

start-virtuoso-mini:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db-mini/virtuoso.ini

start-virtuoso-mini-2:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db-mini-2/virtuoso.ini

start-virtuoso-mini-cai:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db-mini-cai/virtuoso.ini

start-virtuoso-full-cai:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db-full-cai/virtuoso.ini

start-virtuoso-full:
	$(VIRTUOSO_BINARY) -f +configfile local-data/virtuoso-db-full/virtuoso.ini

$(DATA)/mediator-names:
	$(PYTHON) scripts/extract_mediator_names.py > $@
	#grep -f ~/broccoli/freebase/data/freebase.mediators ~/broccoli/freebase/data/name-mapping > $@

#$(DATA)/relation-counts:
#	cat ~/broccoli/freebase/data/freebase.relations-with-subjects-objects-counts.filtered > $@

#
#data/expected-types:
#	python query_sparql.py "PREFIX fb: <http://rdf.freebase.com/ns/> SELECT DISTINCT ?x1 ?x2 WHERE { ?x1 fb:type.property.expected_type ?x2 }"  $(SPARQL_ENDPOINT) > $@

# For scaling out parsing
# Start a central proxy loadbalancing to the parsers
start-haproxy:
	$(HAPROXY_BIN) -f $(HAPROXY_CFG)

# Varnish cashes all HTTP requests in front of Virtuoso.
# Speeds up evaluation.
start-varnish:
	LD_LIBRARY_DIR=$(VARNISH_LIB_DIR) $(VARNISH_BIN) -a $(VARNISH_BIND) -f $(VARNISH_CFG) -F -n /tmp -s malloc,10G -p http_resp_size=10000000 -p http_req_size=1000000 -p http_resp_hdr_len=1000000 -p http_req_hdr_len=1000000

start-varnish-ha:
	LD_LIBRARY_DIR=$(VARNISH_LIB_DIR) $(VARNISH_BIN) -a $(VARNISH_HABIND) -f $(VARNISH_HACFG) -n varnish_ha -s malloc,30G -p http_resp_size=10000000 -p http_req_size=1000000 -p http_resp_hdr_len=1000000 -p http_req_hdr_len=1000000


# Start a set of parsers listening on ports 4000-4004
start-parser-parallel:
	cd corenlp-frontend; ant run-parallel

%_filtered: %
	$(PYPY) scripts/filter_freebase.py $^ > $@

%_mini_filtered: %
	$(PYPY) scripts/filter_freebase.py --mini $^ > $@

publish:
	#rm -rf data_publish
	#mkdir data_publish
	#@echo "Copying data directory and creating new config."
	#python scripts/publish/publish_data.py config.cfg config_publish.cfg data_publish
	#@echo "Adjusting config directories."
	#sed -i 's/data_publish/data/g' config_publish.cfg
	#@echo "Compressing data directory."
	#cd data_publish; tar -zcvf data.tar.gz * --exclude '*.gz'
	@echo "Compressing virtuoso-db."
	tar -zcvf virtuoso.tar.gz virtuoso-db/*





