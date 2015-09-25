"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import logging

logger = logging.getLogger(__name__)


class AnswerType:
    """
    A simple class to represent an ENUM.
    """
    DATE = 1
    OTHER = 2
    CLASS = 3
    SOFT_CLASS = 4

    def __init__(self, type, target_classes=[]):
        self.type = type
        self.target_classes = target_classes

    def as_string(self):
        if self.type == AnswerType.DATE:
            return "Date"
        elif self.type == AnswerType.OTHER:
            return "Other"
        else:
            return ", ".join(self.target_classes)


class AnswerTypeIdentifier:
    """
    A simple classe to identify the target type
    of a query, e.g. 'DATE'
    """

    def __init__(self):
        self.what_type_pos_patterns = [
            # tv programs
            ("NN NNS ", 1, 2),
            # ship class
            ("NN NN ", 1, 2),
            # characters
            ("NNS ", 0, 1),
            # religion
            ("NN ", 0, 1)]
        self.target_class_blacklist = {"sort",
                                       "type",
                                       "kind",
                                       "kinds"}
        self.target_class_date_patterns = {"in what year",
                                           "what year",
                                           "when",
                                           "since when"}

    def starts_with_date_pattern(self, query):
        for p in self.target_class_date_patterns:
            if query.startswith(p):
                return True
        return False

    def identify_target(self, query):
        if self.starts_with_date_pattern(query.query_text):
            query.target_type = AnswerType(AnswerType.DATE)
        elif query.query_text.startswith('where'):
            target_classes = ['location', 'event', 'conference']
            query.target_type = AnswerType(AnswerType.CLASS,
                                           target_classes=target_classes)
            logger.info("Identified query target type: %s" %
                        query.target_type.type)
        elif query.query_text.startswith('who'):
            target_classes = ['person', 'organization', 'employer', 'character']
            query.target_type = AnswerType(AnswerType.CLASS,
                                           target_classes=target_classes)
            logger.info("Identified query target type: %s" %
                        query.target_type.type)
        else:
            query.target_type = AnswerType(AnswerType.OTHER)


