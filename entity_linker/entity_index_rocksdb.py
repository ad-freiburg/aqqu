"""
Provides access to entities via IDs (MIDs) and surface forms (aliases).

Each entity is assigned an ID equivalent to the byte offset in the entity list
file. A hashmap stores a mapping from MID to this offset. Additionally,
another hashmap stores a mapping from surface form to this offset, along with
a score.
Matched entities with additional info (scores, other aliases) are then read
from the list file using the found offset. This avoids keeping all entities
with unneeded info in RAM.

Note: this can be improved in terms of required RAM.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging
from typing import List
import datetime
import config_helper
import rocksdb
from entity_linker.util import normalize_entity_name
from entity_linker.entity_linker import KBEntity

LOG = logging.getLogger(__name__)

CONTROL_PREFIX = b'ctl:'
ETYPES_PREFIX = b'ets:'
ECATEGORY_PREFIX = b'cat:'
ENTITY_PREFIX = b'ent:'
SURFACE_PREFIX = b'srf:'



class ConcatenationMerger(rocksdb.interfaces.AssociativeMergeOperator):
    """
    Merges values by concatenating them without regard to the order of
    concatenation and with a seperator
    """
    def __init__(self, seperator=b''):
        self.seperator = seperator

    def merge(self, key, existing_value, value):
        """
        Merges the existing value with the new value
        """
        if existing_value:
            return (True, existing_value+self.seperator+value)
        return (True, value)

    def name(self):
        """
        Returns the name of this merger
        """
        return b"ConcatenationMerger"


def timestamp():
    """
    Returns the current UTC timestamp as an ISO time string
    encoded as bytes to be stored in the database
    """
    return datetime.datetime.utcnow().isoformat().encode('utf-8')


class EntityIndex:
    """
    A memory based index for finding entities by their surface form,
    their name and their types ordered by relevance.
    """

    def __init__(self,
                 entity_list_file,
                 surface_map_file,
                 entity_index_prefix,
                 entity_types_map_file,
                 entity_category_map_file):
        self.entity_list_file = entity_list_file
        self.surface_map_file = surface_map_file
        self.entity_types_map_file = entity_types_map_file
        self.entity_category_map_file = entity_category_map_file
        self.entity_db = self._get_entity_db(entity_index_prefix)
        LOG.info("Done initializing surface index.")

    def _get_entity_db(self, index_prefix):
        """Return vocabulary by building a new or reading an existing one.

        :param index_prefix:
        :return:
        """
        entity_db_path = index_prefix + "_entity_index.rocksdb"
        entity_db = None
        options = rocksdb.Options()
        options.create_if_missing = True
        options.merge_operator = ConcatenationMerger(b'\t')
        entity_db = rocksdb.DB(entity_db_path,
                               options, read_only=True)
        write_enabled = False
        if not entity_db.get(CONTROL_PREFIX+b'entity_vocab_loaded'):
            if not write_enabled:
                entity_db = rocksdb.DB(entity_db_path, options)
                write_enabled = True
            self._build_entity_vocabulary(entity_db)
        if not entity_db.get(CONTROL_PREFIX+b'entity_types_loaded'):
            if not write_enabled:
                entity_db = rocksdb.DB(entity_db_path, options)
                write_enabled = True
            self._build_entity_types(entity_db)
        if not entity_db.get(CONTROL_PREFIX+b'entity_categories_loaded'):
            if not write_enabled:
                entity_db = rocksdb.DB(entity_db_path, options)
                write_enabled = True
            self._build_entity_categories(entity_db)
        if not entity_db.get(CONTROL_PREFIX+b'surfaces_loaded'):
            if not write_enabled:
                entity_db = rocksdb.DB(entity_db_path, options)
                write_enabled = True
            self._build_surface_index(entity_db)

        if write_enabled:
            # Now we can reopen in read-only mode so as to allow
            # multiple instances to share the same DB
            del entity_db
            entity_db = rocksdb.DB(entity_db_path, options, read_only=True)
        return entity_db

    def _build_surface_index(self, entity_db):
        """
        Build the surface index.

        Reads from the surface map on disk and creates a map from
        surface_form -> offset, score ....
        """
        LOG.info("Building surface index db.")
        num_lines = 0
        with open(self.surface_map_file, 'rb') as in_file:
            for line in in_file:
                num_lines += 1
                cols = line.rstrip().split(b'\t')
                surface_form = cols[0]
                surface_score = cols[1]
                mid = cols[2]
                key = SURFACE_PREFIX+surface_form
                entity_db.merge(key, surface_score+b'\t'+mid)
                if num_lines % 1000000 == 0:
                    LOG.info('Stored %s surface-forms.', num_lines)

        entity_db.put(CONTROL_PREFIX+b'surfaces_loaded', timestamp())

    def _build_entity_types(self, entity_db):
        """
        Create mapping from MID to entity type.
        """
        LOG.info("Building entity -> types db")

        # Remember the offset for each type list
        num_lines = 0
        with open(self.entity_types_map_file, 'rb') as in_file:
            for line in in_file:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    LOG.info('Read %s lines', num_lines)
                cols = line.strip().split(b'\t')
                mid = cols[0]
                entity_db.put(ETYPES_PREFIX+mid, cols[1])
        entity_db.put(CONTROL_PREFIX+b'entity_types_loaded', timestamp())
        LOG.info("Loaded %s entity -> types mappings", num_lines)

    def _build_entity_categories(self, entity_db):
        """
        Create mapping from MID to FreebaseEasy categories
        """
        LOG.info("Building entity -> category db")

        # Remember the offset for each type list
        num_lines = 0
        with open(self.entity_category_map_file, 'rb') as in_file:
            for line in in_file:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    LOG.info('Read %s lines', num_lines)
                cols = line.strip().split(b'\t')
                mid = cols[0]
                if len(cols) < 2:
                    LOG.info('Missing category for: %s', mid)
                    category = b'Unknown'
                else:
                    category = cols[1]
                entity_db.put(ECATEGORY_PREFIX+mid, category)
        entity_db.put(CONTROL_PREFIX+b'entity_categories_loaded', timestamp())
        LOG.info("Loaded %s entity -> category mappings", num_lines)

    def _build_entity_vocabulary(self, entity_db):
        """
        Create mapping from MID to offset/ID.
        """
        LOG.info("Building entity mid vocabulary db")
        num_lines = 0
        # Remember the offset for each entity.
        LOG.info("Reading entity list")
        with open(self.entity_list_file, 'rb') as in_file:
            for line in in_file:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    LOG.info('Read %s lines', num_lines)
                cols = line.strip().split(b'\t')
                mid = cols[0]
                entity_db.put(ENTITY_PREFIX+mid, line)
        entity_db.put(CONTROL_PREFIX+b'entity_vocab_loaded', timestamp())

    @staticmethod
    def init_from_config():
        """Return an instance with options parsed by a config parser.

        :param config_options:
        :return:
        """
        config_options = config_helper.config
        entity_list_file = config_options.get('EntityIndex',
                                              'entity-list')
        entity_surface_map = config_options.get('EntityIndex',
                                                'entity-surface-map')
        entity_index_prefix = config_options.get('EntityIndex',
                                                 'entity-index-prefix')
        entity_types_map = config_options.get('EntityIndex',
                                              'entity-types-map')
        entity_category_map = config_options.get('EntityIndex',
                                                 'entity-category-map')
        return EntityIndex(entity_list_file, entity_surface_map,
                           entity_index_prefix, entity_types_map,
                           entity_category_map)

    def get_entity_for_mid(self, mid: str) -> 'KBEntity':
        """Returns the entity object for the MID or None if the MID is unknown.

        :param mid:
        """
        mid_bytes = mid.encode('utf-8')
        line = self.entity_db.get(ENTITY_PREFIX+mid_bytes)
        if not line:
            LOG.info("No entity for mid: '%s'.", mid)
            return None

        entity = EntityIndex._bytes_to_entity(line)
        return entity

    def get_types_for_mid(self, mid, max_len=None):
        """
        Returns a list of types of the entity with the given mid, ordered
        by relevance on a best effort bases.
        :param mid: mid to look up
        :return: list of types (as strings), if no types are found ['UNK']
        """
        mid = mid.encode('utf-8')
        line = self.entity_db.get(ETYPES_PREFIX+mid)
        if not line:
            LOG.debug("No type known for mid: '%s'.", mid)
            return ['UNK']

        types = EntityIndex._bytes_to_types(line)
        # Note: list[:None] == list[:]
        return types[:max_len]

    def get_category_for_mid(self, mid):
        """
        Returns the FreebaseEasy category for the entity with the given mid
        """
        mid = mid.encode('utf-8')
        category_raw = self.entity_db.get(ECATEGORY_PREFIX+mid)
        if not category_raw:
            LOG.debug("No category known for mid: '%s'.", mid)
            return 'Unknown'

        return category_raw.decode('utf-8')

    def get_entities_for_surface(self, surface):
        """Return all entities for the surface form.

        :param surface:
        :return:
        """
        surface = normalize_entity_name(surface).encode('utf-8')
        LOG.debug("Looking up %s", surface)
        line = self.entity_db.get(SURFACE_PREFIX+surface)
        if not line:
            return []
        cols = line.split(b'\t')

        mids_dedup = set()
        result = []
        for i in range(0, len(cols), 2):
            surface_score = float(cols[i])
            mid = cols[i+1].decode('utf-8')
            if mid in mids_dedup:
                continue
            mids_dedup.add(mid)
            entity = self.get_entity_for_mid(mid)
            if entity:
                result.append((entity, surface_score))
        return result

    @staticmethod
    def _bytes_to_entity(line: bytes) -> 'KBEntity':
        """
        Instantiate entity from string representation.

        >>> e = EntityIndex._bytes_to_entity(b'm.0abc1\\tfoo name\\t7\\tfooly\\tfoo\\n')
        >>> e.name
        'foo name'
        >>> e.id
        'm.0abc1'
        >>> e.score
        7
        >>> e.aliases
        ['fooly', 'foo']
        """
        cols = line.strip().decode('utf-8').split('\t')
        mid = cols[0]
        name = cols[1]
        score = int(cols[2])
        aliases = cols[3:]
        return KBEntity(name, mid, score, aliases)

    @staticmethod
    def _bytes_to_types(line: bytes) -> List[str]:
        """
        Read a list of types from a string. Not that the first column
        currently is the mid

        >>> EntityIndex._bytes_to_types(b'type_a type_b type_c\\n')
        ['type_a', 'type_b', 'type_c']
        """
        types = [t.decode('utf-8') for t in line.strip().split(b' ')]
        return types


if __name__ == '__main__':
    import doctest
    doctest.testmod()
