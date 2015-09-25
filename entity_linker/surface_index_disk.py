"""
DEPRECATED: This is no longer used and will be removed.

Provides access to entities based on their surface form. The index is a large
disk-based hashmap from text -> [entity, entity...] where text consists of
aliases of the entities.
This can be used instead of EntitySurfaceIndexMemory but requires kyotocabinet
to be installed/compiled.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import mmap
import kyotocabinet
import logging
from util import *
import os
import entity_linker
import array

logger = logging.getLogger(__name__)


class EntitySurfaceIndexDisk(object):
    """A disk based index for to find entities matching a surface form."""

    def __init__(self, entity_list_file, surface_map_file,
                 entity_index_prefix):
        surface_index_name = entity_index_prefix + "_surface.kch"
        mid_offset_index_name = entity_index_prefix + "_mid_offsets.kch"
        (index,
         offset_index) = self._get_or_create_index(surface_map_file,
                                                   surface_index_name,
                                                   mid_offset_index_name,
                                                   entity_list_file)
        self.index = index
        self.offset_index = offset_index
        # Memory-map the file.
        self.entities_mm_f = open(entity_list_file, 'r')
        self.entities_mm = mmap.mmap(self.entities_mm_f.fileno(), 0,
                                     prot=mmap.PROT_READ)

    @staticmethod
    def init_from_config(config_options):
        """Return an instance with options parsed by a config parser.

        :param config_options:
        :return:
        """
        entity_list_file = config_options.get('EntitySurfaceIndex',
                                              'entity-list')
        entity_surface_map = config_options.get('EntitySurfaceIndex',
                                                'entity-surface-map')
        entity_index_prefix = config_options.get('EntitySurfaceIndex',
                                                 'entity-index-prefix')
        return EntitySurfaceIndexDisk(entity_list_file, entity_surface_map,
                                      entity_index_prefix)

    def _get_or_create_index(self, surface_map_file, surface_index_name,
                             mid_offset_index_name,
                             entity_list_file):
        """Read existing index or create one if not present.

        :param surface_map_file:
        :param surface_index_name:
        :param mid_offset_index_name:
        :param entity_list_file:
        :return:
        """
        if not os.path.isfile(surface_index_name):
            self._create_index(surface_map_file, entity_list_file,
                               surface_index_name, mid_offset_index_name)

        surface_db = kyotocabinet.DB()
        if not surface_db.open(surface_index_name,
                               kyotocabinet.DB.OREADER |
                               kyotocabinet.DB.ONOREPAIR):
            logger.error('Error opening entities surface index.')
        mid_offset_db = kyotocabinet.DB()
        if not mid_offset_db.open(mid_offset_index_name,
                                  kyotocabinet.DB.OREADER |
                                  kyotocabinet.DB.ONOREPAIR):
            logger.error('Error opening entities surface index.')
        # db = yakc.KyotoDB(index_name, type='HashDB')
        logging.info("Reusing entities surface index with %s keys."
                     % (surface_db.count()))
        return surface_db, mid_offset_db

    def _create_index(self, surface_map_file, entity_list_file,
                      surface_index_name, mid_offset_index_name):
        logging.info("Generating entities and surface index.")
        num_lines = 0
        logger.info("Reading entity offsets.")
        mid_offsets = dict()
        # Remember the offset for each entity.
        with open(entity_list_file, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            offset = mm.tell()
            line = mm.readline()
            while line:
                num_lines += 1
                if num_lines % 1000000 == 0:
                    logger.info('Read %s lines' % num_lines)
                line = line.decode('utf-8')
                cols = line.split('\t')
                mid = cols[0]
                mid_offsets[mid] = offset
                offset = mm.tell()
                line = mm.readline()
        s_index_db = kyotocabinet.DB()
        s_index_db.open(
            surface_index_name + '#msiz=20000000000#bnum=200000000#opts=l')
        logging.info("Creating surface map on disk.")
        num_lines = 0
        # We now write a list of (offset, score)... floats for
        # each surface form.
        num_not_found = 0
        with open(surface_map_file, 'r') as f:
            last_surface_form = None
            surface_form_entries = array.array('d')
            for line in f:
                num_lines += 1
                try:
                    cols = line.decode('utf-8').split('\t')
                    surface_form = cols[0]
                    score = float(cols[1])
                    mid = cols[2].strip()
                    offset = float(mid_offsets[mid])
                    if surface_form != last_surface_form:
                        if surface_form_entries:
                            s_index_db.set(last_surface_form,
                                           surface_form_entries.tostring())
                        last_surface_form = surface_form
                        surface_form_entries = array.array('d')
                    surface_form_entries.append(offset)
                    surface_form_entries.append(score)
                except KeyError:
                    num_not_found += 1
                    if num_not_found < 100:
                        logger.warn(
                            "Mid %s appears in surface map but not "
                            "in entity list." % mid)
                    elif num_not_found == 100:
                        logger.warn(
                            "Suppressing further warnings about unfound mids.")
                if num_lines % 1000000 == 0:
                    logger.info(
                        'Stored %s surface-form->entity pairs.' % num_lines)
            if surface_form_entries:
                s_index_db.set(last_surface_form,
                               surface_form_entries.tostring())
        if num_not_found > 0:
            logger.warn(
                "%s entries of an mid in surface map but mid not "
                "in entity list." % num_not_found)
        # store an additional index from mid -> offset
        s_index_db.close()
        mid_offset_db = kyotocabinet.DB()
        mid_offset_db.open(
            mid_offset_index_name + '#msiz=20000000000#bnum=200000000#opts=l')
        logging.info("Creating entity offset index on disk.")
        for mid, offset in mid_offsets.iteritems():
            mid_offset_db.set(mid, offset)
        logging.info("Done.")
        mid_offset_db.close()

    def get_entity_for_mid(self, mid):
        """Returns the entity object for the MID or None if the MID is unknown.

        :param mid:
        :return:
        """
        offset = self.offset_index[mid]
        entity = None
        if offset:
            entity = self._read_entity_from_offset(int(offset))
        else:
            logger.warn("Unknown entity mid: '%s'." % mid)
        return entity

    def get_entities_for_surface(self, surface):
        """Return all entities that match the surface form.

        Returns list of tuples of entity and surface_score

        :param surface:
        :return:
        """

        surface = normalize_entity_name(surface)
        ids = self.index.get(surface)
        if ids is None:
            return []
        else:
            ids_array = array.array('d')
            ids_array.fromstring(ids)
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

    def _string_to_entity(self, line):
        """Instantiate entity from string based representation.

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
        """Read string based representation from disk offset.

        :param offset:
        :return:
        """
        self.entities_mm.seek(offset)
        l = self.entities_mm.readline()
        return self._string_to_entity(l)
