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
import mmap
import logging
import pickle
from collections import defaultdict
import marshal
import os
from .util import normalize_entity_name
import array
from . import entity_linker
import globals

logger = logging.getLogger(__name__)

# TODO(schnelle) marshal doesn't support class types, but pickle is super
# super slow so instead of using this nice class we have to resolve to tuples
#class MIDOffsets:
#    """
#    Holds offset information associated with an MID, using __slots__
#    immensely reduces memory consumption
#    """
#    __slots__ = ['entity_offset', 'types_offset']
#    def __init__(self, entity_offset=None, types_offset=None):
#        self.entity_offset = entity_offset
#        self.types_offset = types_offset
ENTITY_OFFSET = 0
TYPES_OFFSET = 1

class EntityIndex(object):
    """
    A memory based index for finding entities by their surface form,
    their name and their types ordered by relevance.
    """

    def __init__(self,
                 entity_list_file,
                 surface_map_file,
                 entity_index_prefix_file,
                 entity_types_map_file):
        self.entity_list_file = entity_list_file
        self.surface_map_file = surface_map_file
        self.entity_types_map_file = entity_types_map_file
        self.mid_vocabulary = self._get_entity_vocabulary(entity_index_prefix_file)
        self.surface_index = self._get_surface_index(entity_index_prefix_file)
        self.types_mm_f = open(entity_types_map_file, 'rb')
        self.types_mm = mmap.mmap(self.types_mm_f.fileno(), 0,
                                     prot=mmap.PROT_READ)
        self.entities_mm_f = open(entity_list_file, 'rb')
        self.entities_mm = mmap.mmap(self.entities_mm_f.fileno(), 0,
                                     prot=mmap.PROT_READ)
        logger.info("Done initializing surface index.")

    def _get_entity_vocabulary(self, index_prefix):
        """Return vocabulary by building a new or reading an existing one.

        :param index_prefix:
        :return:
        """
        vocab_file = index_prefix + "_mid_vocab"
        if os.path.isfile(vocab_file):
            logger.info("Loading entity vocabulary from disk.")
            vocabulary = marshal.load(open(vocab_file, 'rb'))
        else:
            vocabulary = self._build_entity_vocabulary()
            logger.info("Writing entity vocabulary to disk.")
            marshal.dump(vocabulary, open(vocab_file, 'wb'))
        return vocabulary

    def _get_surface_index(self, index_prefix):
        """Return surface index by building new or reading existing one.

        :param index_prefix:
        :return:
        """
        surface_index_file = index_prefix + "_surface_index"
        if os.path.isfile(surface_index_file):
            logger.info("Loading surfaces from disk.")
            surface_index = marshal.load(open(surface_index_file, 'rb'))
        else:
            surface_index = self._build_surface_index()
            logger.info("Writing entity surfaces to disk.")
            marshal.dump(surface_index, open(surface_index_file, 'wb'))
        return surface_index

    def _build_surface_index(self):
        """Build the surface index.

        Reads from the surface map on disk and creates a map from
        surface_form -> offset, score ....

        :return:
        """
        n_lines = 0
        surface_index = dict()
        num_not_found = 0
        with open(self.surface_map_file, 'rb') as f:
            last_surface_form = None
            surface_form_entries = array.array('d')
            for line in f:
                n_lines += 1
                cols = line.rstrip().split(b'\t')
                surface_form = cols[0]
                score = float(cols[1])
                mid = cols[2]
                try:
                    entity_offset = self.mid_vocabulary[mid][ENTITY_OFFSET]
                except KeyError:
                    num_not_found += 1
                    if num_not_found < 100:
                        logger.warn("Mid %s appears in surface map but "
                                    "not in entity list." % cols[2])
                    elif num_not_found == 100:
                        logger.warn("Suppressing further warnings about "
                                    "unfound mids.")
                    continue
                if entity_offset == None:
                    logger.warn("mid %s has type offset but no entity offset"%mid)
                    continue

                if surface_form != last_surface_form:
                    if surface_form_entries:
                        surface_index[
                            last_surface_form] = surface_form_entries
                    last_surface_form = surface_form
                    surface_form_entries = array.array('d')
                #TODO(schnelle) saving an entity_id in a double
                #               is suuuper ugly
                surface_form_entries.append(float(entity_offset))
                surface_form_entries.append(score)
                if n_lines % 1000000 == 0:
                    logger.info('Stored %s surface-forms.' % n_lines)

            if surface_form_entries:
                surface_index[last_surface_form] = surface_form_entries
        logger.warn("%s entity appearances in surface map w/o mapping to "
                    "entity list" % num_not_found)
        return surface_index

    def _build_entity_vocabulary(self):
        """Create mapping from MID to offset/ID.

        :return:
        """
        logger.info("Building entity mid vocabulary.")
        mid_vocab = dict()
        num_lines = 0
        # Remember the offset for each entity.
        logger.info("Reading entity list")
        with open(self.entity_list_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            entity_offset = mm.tell()
            line = mm.readline()
            while line:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    logger.info('Read %s lines' % num_lines)
                cols = line.strip().split(b'\t')
                mid = cols[0]
                # in a defaultdict access always works
                mid_vocab[mid] = (entity_offset, None)
                entity_offset = mm.tell()
                line = mm.readline()

        # Remember the offset for each type list
        logger.info("Reading entity -> types map")
        with open(self.entity_types_map_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            types_offset = mm.tell()
            line = mm.readline()
            while line:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    logger.info('Read %s lines' % num_lines)
                cols = line.strip().split(b'\t')
                mid = cols[0]
                # in a defaultdict access always works
                entity_offset = mid_vocab[mid][ENTITY_OFFSET] \
                        if mid in mid_vocab else None
                mid_vocab[mid] = (entity_offset, types_offset)
                types_offset = mm.tell()
                line = mm.readline()
        return mid_vocab

    @staticmethod
    def init_from_config():
        """Return an instance with options parsed by a config parser.

        :param config_options:
        :return:
        """
        config_options = globals.config
        entity_list_file = config_options.get('EntityIndex',
                                              'entity-list')
        entity_surface_map = config_options.get('EntityIndex',
                                                'entity-surface-map')
        entity_index_prefix = config_options.get('EntityIndex',
                                                 'entity-index-prefix')
        entity_types_map = config_options.get('EntityIndex',
                                                      'entity-types-map')
        return EntityIndex(entity_list_file, entity_surface_map,
                                        entity_index_prefix, entity_types_map)

    def get_entity_for_mid(self, mid):
        """Returns the entity object for the MID or None if the MID is unknown.

        :param mid:
        :return:
        """
        mid = mid.encode('utf-8')
        try:
            offset = self.mid_vocabulary[mid][ENTITY_OFFSET]
        except KeyError:
            offset = None

        if offset == None:
            logger.warn("No type entity for mid: '%s'." % mid)
            return ['UNK']

        entity = self._read_entity_from_offset(int(offset))
        return entity

    def get_types_for_mid(self, mid, max_len=None):
        """
        Returns a list of types of the entity with the given mid, ordered
        by relevance on a best effort bases.
        :param mid: mid to look up
        :return: list of types (as strings), if no types are found ['UNK']
        """
        mid = mid.encode('utf-8')
        try:
            offset = self.mid_vocabulary[mid][TYPES_OFFSET]
        except KeyError:
            offset = None

        if offset == None:
            logger.warn("No type known for mid: '%s'." % mid)
            return ['UNK']


        types = self._read_types_from_offset(offset)
        # TODO(schnelle) we may want to only read max_len entries
        # Note: list[:None] == list[:]
        return types[:max_len]

    def get_entities_for_surface(self, surface):
        """Return all entities for the surface form.

        :param surface:
        :return:
        """
        surface = normalize_entity_name(surface)
        surface = surface.encode('utf-8')
        try:
            bytestr = self.surface_index[surface]
            ids_array = array.array('d')
            ids_array.fromstring(bytestr)
            result = []
            i = 0
            while i < len(ids_array) - 1:
                offset = ids_array[i]
                surface_score = ids_array[i + 1]
                entity = self._read_entity_from_offset(int(offset))
                # Check if the main name of the entity exactly matches the text.
                result.append((entity, surface_score))
                i += 2
            return result
        except KeyError:
            return []

    @staticmethod
    def _string_to_entity(line):
        """Instantiate entity from string representation.

        :param line:
        :return:
        """
        line = line.decode('utf-8')
        cols = line.strip().split('\t')
        mid = cols[0]
        name = cols[1]
        score = int(cols[2])
        aliases = cols[3:]
        return entity_linker.KBEntity(name, mid, score, aliases)

    def _read_entity_from_offset(self, offset):
        """Read entity string representation from offset.

        :param offset:
        :return:
        """
        self.entities_mm.seek(offset)
        l = self.entities_mm.readline()
        return self._string_to_entity(l)

    def _read_types_from_offset(self, offset):
        """
        Read a list of types from the given offset. Note that the
        offset points to the whole line beginning with the mid
        """
        self.types_mm.seek(offset)
        line = self.types_mm.readline()
        _, types_all = line.split(b'\t')
        types = [t.decode('utf-8') for t in types_all.strip().split(b' ')]
        return types



def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(module)s : %(message)s',
        level=logging.INFO)
    index = EntityIndex('data/entity-list_cai',
                                     'data/entity-surface-map_cai',
                                     'iprefix')
    print(index.get_entities_for_surface("albert einstein"))


if __name__ == '__main__':
    main()
