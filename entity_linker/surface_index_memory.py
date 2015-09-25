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
import os
from util import normalize_entity_name
import array
import marshal
import entity_linker
import globals

logger = logging.getLogger(__name__)


class EntitySurfaceIndexMemory(object):
    """A memory based index for finding entities."""

    def __init__(self,
                 entity_list_file,
                 surface_map_file,
                 entity_index_prefix):
        self.entity_list_file = entity_list_file
        self.surface_map_file = surface_map_file
        self.mid_vocabulary = self._get_entity_vocabulary(entity_index_prefix)
        self.surface_index = self._get_surface_index(entity_index_prefix)
        self.entities_mm_f = open(entity_list_file, 'r')
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
            vocabulary = marshal.load(open(vocab_file, 'r'))
        else:
            vocabulary = self._build_entity_vocabulary()
            logger.info("Writing entity vocabulary to disk.")
            marshal.dump(vocabulary, open(vocab_file, 'w'))
        return vocabulary

    def _get_surface_index(self, index_prefix):
        """Return surface index by building new or reading existing one.

        :param index_prefix:
        :return:
        """
        surface_index_file = index_prefix + "_surface_index"
        if os.path.isfile(surface_index_file):
            logger.info("Loading surfaces from disk.")
            surface_index = marshal.load(open(surface_index_file, 'r'))
        else:
            surface_index = self._build_surface_index()
            logger.info("Writing entity surfaces to disk.")
            marshal.dump(surface_index, open(surface_index_file, 'w'))
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
        with open(self.surface_map_file, 'r') as f:
            last_surface_form = None
            surface_form_entries = array.array('d')
            for line in f:
                n_lines += 1
                try:
                    cols = line.rstrip().split('\t')
                    surface_form = cols[0]
                    score = float(cols[1])
                    mid = cols[2]
                    entity_id = self.mid_vocabulary[mid]
                    if surface_form != last_surface_form:
                        if surface_form_entries:
                            surface_index[
                                last_surface_form] = surface_form_entries
                        last_surface_form = surface_form
                        surface_form_entries = array.array('d')
                    surface_form_entries.append(entity_id)
                    surface_form_entries.append(score)
                except KeyError:
                    num_not_found += 1
                    if num_not_found < 100:
                        logger.warn("Mid %s appears in surface map but "
                                    "not in entity list." % cols[2])
                    elif num_not_found == 100:
                        logger.warn("Suppressing further warnings about "
                                    "unfound mids.")
                if n_lines % 1000000 == 0:
                    logger.info('Stored %s surface-forms.' % n_lines)
            if surface_form_entries:
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
        with open(self.entity_list_file, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            offset = mm.tell()
            line = mm.readline()
            while line:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    logger.info('Read %s lines' % num_lines)
                cols = line.strip().split('\t')
                mid = cols[0]
                mid_vocab[mid] = offset
                offset = mm.tell()
                line = mm.readline()
        return mid_vocab

    @staticmethod
    def init_from_config():
        """Return an instance with options parsed by a config parser.

        :param config_options:
        :return:
        """
        config_options = globals.config
        entity_list_file = config_options.get('EntitySurfaceIndex',
                                              'entity-list')
        entity_surface_map = config_options.get('EntitySurfaceIndex',
                                                'entity-surface-map')
        entity_index_prefix = config_options.get('EntitySurfaceIndex',
                                                 'entity-index-prefix')
        return EntitySurfaceIndexMemory(entity_list_file, entity_surface_map,
                                        entity_index_prefix)

    def get_entity_for_mid(self, mid):
        """Returns the entity object for the MID or None if the MID is unknown.

        :param mid:
        :return:
        """
        try:
            offset = self.mid_vocabulary[mid]
            entity = self._read_entity_from_offset(int(offset))
            return entity
        except KeyError:
            logger.warn("Unknown entity mid: '%s'." % mid)
            return None

    def get_entities_for_surface(self, surface):
        """Return all entities for the surface form.

        :param surface:
        :return:
        """
        surface = normalize_entity_name(surface)
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


def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(module)s : %(message)s',
        level=logging.INFO)
    index = EntitySurfaceIndexMemory('data/entity-list_cai',
                                     'data/entity-surface-map_cai',
                                     'iprefix')
    print index.get_entities_for_surface("Albert Einstein")


if __name__ == '__main__':
    main()
