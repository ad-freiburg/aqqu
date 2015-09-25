"""
Interface to perform NLP tasks using a remote corenlp instance.

Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import re
from json import loads
import logging
import urllib
import urllib2
import time
from collections import defaultdict
import globals

logger = logging.getLogger(__name__)


# The delimiter in the serialized string based representation of a dependency.
DEP_PARSE_TOKEN_SPLIT = "-"


class CoreNLPParser(object):

    def __init__(self, host):
        self.host = host

    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        parser_host = config_options.get('CoreNLPParser', 'host')
        return CoreNLPParser(parser_host)

    def parse(self, query, include_json=False,
              parse_trees=False):
        """Parse some text and return ParseResult.

        :param query:
        :param include_json: Whether to include the JSON string in the
         Result object.
        :param parse_trees: Whether to create syntactic parse tree
        representations.
        :return:
        """
        logger.debug(u"Parsing '{0}'.".format(query))
        t0 = time.time()
        value = {"text": query.encode('utf-8')}
        data = urllib.urlencode(value)
        req = urllib2.Request(self.host, data)
        response = urllib2.urlopen(req).read()
        parse_result = parse_json_result(response,
                                         include_json=include_json,
                                         parse_trees=parse_trees)
        duration = (time.time() - t0) * 1000
        logger.debug("Parse took %.2f ms." % duration)
        return parse_result


def extract_tokens_from_json(result):
    """Create list of token objects from JSON result.

    :param result:
    :return:
    """
    tokens = []
    for i, w in enumerate(result['words']):
        t = Token(w["token"])
        t.index = i
        t.pos = w['pos']
        t.lemma = w['lemma']
        if 'norm-ner' in w:
            t.nne_tag = w['norm-ner']
        if 'ner' in w:
            t.ne_tag = w['ner']
        tokens.append(t)
    return tokens


def extract_parses_from_json(result, tokens, parse_trees=False):
    """Create syntactic parse results from JSON result.

    If parse_trees is True, create tree and graph representations.
    :param result:
    :param tokens:
    :return:
    """
    parse = None
    dep_parse = None
    if 'tree' in result:
        parse = ConstituentParse(result['tree'])
        if parse_trees:
            parse.tree = parse.build_constituent_parse_tree()
    if 'dependencies' in result:
        dep_parse = DependencyParse(result['dependencies'])
        if parse_trees:
            dep_parse.graph = dep_parse.build_dependency_graph(tokens)
    return parse, dep_parse


def parse_json_result(json_response, include_json=False,
                      parse_trees=False):
    """Create the ParseResult from the returned JSON."""
    result = loads(json_response, encoding='utf-8')
    # ujson is faster!
    # result = ujson.loads(json_response)
    tokens = extract_tokens_from_json(result)
    c_parse, d_parse = extract_parses_from_json(result, tokens,
                                                parse_trees=parse_trees)
    parse_result = ParseResult(tokens, c_parse, d_parse)
    if include_json:
        parse_result.json_result = json_response.decode('utf-8')
    return parse_result


class ParseResult(object):
    """Result of a parse including tokens and syntactic parses."""

    def __init__(self, tokens, constituent_parse,
                 dependency_parse):
        self.tokens = tokens
        self.constituent_parse = constituent_parse
        self.dependency_parse = dependency_parse
        self.json_result = None


class Token(object):
    """A single Token processed by the parser."""

    def __init__(self, token):
        self.token = token
        self.pos = None
        self.lemma = None
        self.index = None
        self.ne_tag = None
        self.nne_tag = None
        self.index = None
        # A link to the dependency graph node.
        self.dep_node = None


class DependencyParse(object):
    """The dependency parse result."""

    def __init__(self, dependencies):
        self.graph = None
        self.dependencies = [tuple(x) for x in dependencies]

    def build_dependency_graph(self, tokens):
        """Build a graph representation from the dependencies.

        :param tokens:
        :return:
        """
        t0 = time.time()
        nodes = {}
        for (rel, head, dep) in self.dependencies:
            if head not in nodes:
                node = DependencyNode(head)
                if tokens and node.position != 0:
                    node.token = tokens[node.position - 1]
                    node.token.dep_node = node
                nodes[head] = node
            if dep not in nodes:
                node = DependencyNode(dep)
                if tokens and node.position != 0:
                    node.token = tokens[node.position - 1]
                    node.token.dep_node = node
                nodes[dep] = node
            nodes[head].add_dependant(nodes[dep], rel)
        # Construct graph object.
        if "ROOT-0" in nodes:
            root = nodes["ROOT-0"]
        else:
            root = nodes.items()[0]
        g = DependencyGraph(nodes.values(), root)
        duration = (time.time() - t0) * 1000
        logger.debug("Building dependency graph took %s ms." % duration)
        return g


class ConstituentParse(object):
    """The constituent parse result."""

    def __init__(self, parse_string):
        self.parse_string = parse_string
        self.parse_tree = None

    def build_constituent_parse_tree(self):
        """Build a parse tree from its string representation.

        :return:
        """
        t0 = time.time()
        parse_str = self.parse_string
        parse_tokens = filter(lambda x: x != '' and x != ' ',
                              re.split(r'([ \)])', parse_str))
        root = ConstituentNode(parse_tokens[0][1:])
        nodes = []
        parent_nodes = [root]
        word_position = 0
        for t in parse_tokens[1:]:
            if t[0] == '(':
                node = ConstituentNode(t[1:])
                node.start_position = word_position
                nodes.append(node)
                parent_nodes[-1].children.append(node)
                parent_nodes.append(node)
            elif t == ')':
                parent_nodes[-1].end_position = word_position - 1
                parent_nodes.pop()
            else:
                node = ConstituentNode(t)
                node.start_position = word_position
                node.end_position = word_position
                nodes.append(node)
                node.is_terminal = True
                parent_nodes[-1].is_preterminal = True
                parent_nodes[-1].children.append(node)
                word_position += 1
        # All parents should be pop-ed here.
        assert(len(parent_nodes) == 0)
        duration = (time.time() - t0) * 1000
        logger.debug("Building constituent tree took %s ms." % duration)
        return ConstituentTree(root, nodes)


class ConstituentTree(object):
    """The constituent tree.

    Contains with a pointer to the root and all containing nodes.
    """

    def __init__(self, root, nodes):
        self.root = root
        self.nodes = nodes


class ConstituentNode(object):
    """A single node in the constituent tree.

    Can be a leaf or internal node and contains links to  its children.
    """

    def __init__(self, tag):
        self.tag = tag
        self.children = []
        self.is_terminal = False
        self.is_preterminal = False
        self.start_position = 0
        self.end_position = 0

    def print_indent(self, indent=' '):
        print indent + self.tag + '(%s, %s)' % (self.start_position,
                                                self.end_position)
        for c in self.children:
            c.print_indent(indent=indent + ' ')

    def all_children_preterminal(self):
        return all([c.is_preterminal for c in self.children])


class DependencyGraph:
    """The dependency graph with a pointer to its root and containing nodes."""

    def __init__(self, nodes, root):
        self.nodes = nodes
        self.root = root

    def print_dependencies(self):
        for n in self.nodes:
            if n.heads:
                for head in n.heads:
                    print "({}, {}, {}) {}".format(n.heads_rel[head],
                                                   head.word,
                                                   n.word,
                                                   n.token.pos)

    def shortest_path(self, node_a, node_b):
        """Perform simple Dijkstra.

        Finds shortest path between two nodes.
        Returns list of nodes on path including source and target.
        """
        dist = {}
        prev = {}
        max_dist = 999
        # Because the python prio-queue has no decrease key
        # operation we simply use a dict
        queue = {}
        dist[node_a] = 0
        for n in self.nodes:
            if n != node_a:
                dist[n] = max_dist
                prev[n] = None
            queue[n] = dist[n]
        while queue:
            u, d_u = DependencyGraph._min_queue(queue)
            del queue[u]
            if u == node_b:
                break
            for v in u.dependants + u.heads:
                d = d_u + 1
                if d < dist[v]:
                    dist[v] = d
                    prev[v] = u
                    queue[v] = d
        # It could happen, that there is no path at all.
        if dist[node_b] < max_dist:
            # Reconstruct path
            path = [node_b]
            p = prev[node_b]
            while p != node_a:
                path.append(p)
                p = prev[p]
            path.append(p)
            path.reverse()
            return path
        else:
            # Return empty path
            return []

    @staticmethod
    def _min_queue(queue):
        """Pop min from queue."""
        minkey = None
        minvalue = 99999
        for key, value in queue.iteritems():
            if value < minvalue:
                minkey = key
                minvalue = value
        return minkey, minvalue


class DependencyNode:
    """A single node in the dependency graph with its connected nodes."""

    def __init__(self, name):
        # The name of this node.
        self.name = name
        # The position of the word of this node.
        self.position = int(name[name.rindex(DEP_PARSE_TOKEN_SPLIT) + 1:])
        # The word of this node.
        self.word = name[:name.rindex(DEP_PARSE_TOKEN_SPLIT)]
        # List of all dependants. Can be empty.
        self.dependants = []
        # List of this nodes heads. Can be none for ROOT node.
        self.heads = []
        # A map from node -> head: (head node with this rel)
        self.heads_rel = dict()
        # A map from node -> head: (dep node with this rel)
        self.deps_rel = dict()
        # A map from rel -> nodes: (head nodes with this rel)
        self.rel_heads = defaultdict(list)
        # A map from rel -> nodes: (dep nodes with this rel)
        self.rel_deps = defaultdict(list)
        # Part of Speech tag of the word attached to this node.
        self.token = None

    def add_dependant(self, dep, rel="x"):
        """Add dependant to this node with dependency relation rel.

        :param dep:
        :param rel:
        :return:
        """
        self.dependants.append(dep)
        self.deps_rel[dep] = rel
        self.rel_deps[rel].append(dep)
        # noinspection PyProtectedMember
        dep._add_head(self, rel)

    def _add_head(self, head, rel="x"):
        self.heads.append(head)
        self.heads_rel[head] = rel
        self.rel_heads[rel].append(head)


def main():
    for query in ['He likes the first 13 american states from the last year']:
        parser = CoreNLPParser("http://localhost:10001/parse")
        parse_result = parser.parse(query, parse_trees=True)
        parse_result.dependency_parse.graph.print_dependencies()
        source = parse_result.dependency_parse.graph.nodes[0]
        target = parse_result.dependency_parse.graph.nodes[10]
        path = parse_result.dependency_parse.graph.shortest_path(source, target)
        for p in path:
            print p.token.token

if __name__ == '__main__':
    main()
