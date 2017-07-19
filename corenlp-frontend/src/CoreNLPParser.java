import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.parser.shiftreduce.ShiftReduceParser;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphFactory;
import edu.stanford.nlp.time.TimeAnnotator;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Logger;

/**
 * Created by haussmae on 26.11.2014.
 * A simple Wrapper around the CoreNLP annotators. Currently
 * parser and NER are not implemented.
 */
public class CoreNLPParser {

    /**
     * Logging instance.
     */
    private static final Logger LOGGER =
            Logger.getLogger(CoreNLPServer.class.getName());

    /**
     * The default parser model.
     * DefaultPaths.DEFAULT_PARSER_MODEL
     */
    //private static final String PARSE_MODEL = "edu/stanford/nlp/models/lexparser/englishRNN.ser.gz";
    private static final String PARSE_MODEL = "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz";
    //private static final String PARSE_MODEL = DefaultPaths.DEFAULT_PARSER_MODEL;
    //

    private static final String POSTAG_MODEL = "edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger";
    //private static final String POSTAG_MODEL = DefaultPaths.DEFAULT_POS_MODEL;

    private static final String NER_MODEL = DefaultPaths.DEFAULT_NER_THREECLASS_MODEL;


    /**
     * The maximum length of sentences to parse.
     */
    private static final int MAX_SENT_LENGTH = 200;

    /**
     * An instance of the parser.
     */
    // private final LexicalizedParser lp;

    /**
     * Tokenizer annotator instance.
     */
    final TokenizerAnnotator tokenizerAnnotator;

    /**
     * Words to sentences annotator.
     */
    final WordsToSentencesAnnotator sentenceAnnotator;

    /**
     * A parser annotator annotates the parse tags.
     */
    //final ParserAnnotator parserAnnotator;

    /**
     * A part of speech annotator.
     */
    final POSTaggerAnnotator posAnnotator;


    /**
     * A part of speech annotator.
     */
    final MorphaAnnotator morphAnnotator;

    /**
     * An annotator for time expressions.
     */
    // final TimeAnnotator timeAnnotator;

    /**
     * A named entity annotator.
     */
    private NERCombinerAnnotator nerAnnotator;


    private final DependencyParser dependencyParser;

    /**
     * Default constructr.
     */
    public CoreNLPParser() throws IOException, ClassNotFoundException {
        // TODO Auto-generated constructor stub
        //this.lp = LexicalizedParser.loadModel(PARSE_MODEL);
        //this.parserAnnotator = new ParserAnnotator(ShiftReduceParser.loadModel(PARSE_MODEL), false, MAX_SENT_LENGTH);
        this.sentenceAnnotator = new WordsToSentencesAnnotator(false);
        this.morphAnnotator = new MorphaAnnotator();
        this.tokenizerAnnotator = new TokenizerAnnotator();
        Properties dependencyProps = new Properties();
        dependencyProps.setProperty("testThreads", "1");
        dependencyProps.setProperty("sentenceTimeout", "0");
        this.dependencyParser = DependencyParser.loadFromModelFile(DependencyParser.DEFAULT_MODEL);
        // this.dependencyAnnotator = new DependencyParseAnnotator(dependencyProps);
        // this.timeAnnotator = new TimeAnnotator("sutime", props);
        this.posAnnotator = new POSTaggerAnnotator(POSTAG_MODEL, false);
    }

    private void initializeNER() {
        Properties props = new Properties();
        // These two properties are needed to enable identification of time ranges.
        props.setProperty("sutime.markTimeRanges", "false");
        props.setProperty("sutime.includeRange", "false");
        this.nerAnnotator = new NERCombinerAnnotator(NER_MODEL, props);
    }

    private NERCombinerAnnotator getNERAnnotator() {
      if (this.nerAnnotator == null) {
        this.initializeNER();
      }
      return this.nerAnnotator;
    }

    // Annotate the whole text as a single sentence.
    private void annotateSentence(Annotation document) {
        List<CoreMap> sentences = new ArrayList<CoreMap>();
        List<CoreLabel> tokens = document.get(CoreAnnotations.TokensAnnotation.class);
        if (tokens.size() < 1) {
            document.set(CoreAnnotations.SentencesAnnotation.class, sentences);
            return;
        }
        String text = document.get(CoreAnnotations.TextAnnotation.class);
        int begin = tokens.get(0).get(CoreAnnotations.CharacterOffsetBeginAnnotation.class);
        int last = tokens.size() - 1;
        int end = tokens.get(last).get(CoreAnnotations.CharacterOffsetEndAnnotation.class);
        String sentenceText = text.substring(begin, end);
        Annotation sentence = new Annotation(sentenceText);
        sentence.set(CoreAnnotations.CharacterOffsetBeginAnnotation.class, begin);
        sentence.set(CoreAnnotations.CharacterOffsetEndAnnotation.class, end);
        sentence.set(CoreAnnotations.TokensAnnotation.class, tokens);
        sentence.set(CoreAnnotations.TokenBeginAnnotation.class, 0);
        sentence.set(CoreAnnotations.TokenEndAnnotation.class, last);
        sentence.set(CoreAnnotations.SentenceIndexAnnotation.class, sentences.size());
        for (int i = 0; i < tokens.size(); ++i) {
            CoreLabel token = tokens.get(i);
            token.set(CoreAnnotations.SentenceIndexAnnotation.class, sentences.size());
            token.setIndex(i + 1);
        }
        sentences.add(sentence);
        document.set(CoreAnnotations.SentencesAnnotation.class, sentences);
    }

    public Annotation parse(String sentence, boolean perfomNER) {
        long start = System.currentTimeMillis();
        LOGGER.info("Parsing " + sentence);
        Annotation document = new Annotation(sentence);
        tokenizerAnnotator.annotate(document);
        annotateSentence(document);
        posAnnotator.annotate(document);
        morphAnnotator.annotate(document);
        // Use the NN based dependency parser now.
        //parserAnnotator.annotate(document);
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        for (CoreMap sentenceMap : sentences) {
            GrammaticalStructure gs = this.dependencyParser.predict(sentenceMap);
           SemanticGraph uncollapsedDeps = SemanticGraphFactory.makeFromTree(gs, SemanticGraphFactory.Mode.BASIC, GrammaticalStructure.Extras.NONE, true, null);
            sentenceMap.set(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class, uncollapsedDeps);
        }
        if (perfomNER) {
            NERCombinerAnnotator nerAnnotator = getNERAnnotator();
            nerAnnotator.annotate(document);
        }
        long duration = System.currentTimeMillis() - start;
        LOGGER.info("Parsing took " + duration + " ms.");
        return document;
    }

    public void parseAnnotationToJson(Annotation document) {
        List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
        CoreMap parsedSentence = sentences.get(0);
        String text = parsedSentence.get(CoreAnnotations.TextAnnotation.class);
        //Tree tree = parsedSentence.get(TreeCoreAnnotations.TreeAnnotation.class);
        //String parseTree = tree.toString();
        SemanticGraph g = parsedSentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
        List<Dependency> dependencies = dependenciesFromGraph(g);
        List<CoreLabel> tokens = document.get(CoreAnnotations.TokensAnnotation.class);
        for (CoreLabel t : tokens) {
            System.out.println(t.get(CoreAnnotations.TextAnnotation.class));
            System.out.println(t.get(CoreAnnotations.PartOfSpeechAnnotation.class));
            System.out.println(t.get(CoreAnnotations.LemmaAnnotation.class));
            System.out.println(t.get(CoreAnnotations.NamedEntityTagAnnotation.class));
            System.out.println(t.get(CoreAnnotations.CharacterOffsetBeginAnnotation.class));
            System.out.println(t.get(CoreAnnotations.CharacterOffsetEndAnnotation.class));
        }
    }

    private class Dependency {

        public String relation;
        public String head;
        public String dependant;

        public Dependency(String relation, String head, String dependant) {
            this.relation = relation;
            this.head = head;
            this.dependant = dependant;
        }
    }

    /**
     * Returns a list of dependencies. Each dependency is
     * relation, head, dependant.
     *
     * @param g
     * @return
     */
    private List<Dependency> dependenciesFromGraph(SemanticGraph g) {
        ArrayList<Dependency> dependencies = new ArrayList<>();
        StringBuilder buf = new StringBuilder();
        for (IndexedWord root : g.getRoots()) {
            dependencies.add(new Dependency("root", "ROOT-0", toDepStyle(root)));
        }
        for (SemanticGraphEdge edge : g.edgeListSorted()) {
            dependencies.add(new Dependency(edge.getRelation().toString(),
                    toDepStyle(edge.getSource()), toDepStyle(edge.getTarget())));
        }
        return dependencies;
    }

    public List<String[]> dependencyStrings(CoreMap parsedSentence) {
        SemanticGraph g = parsedSentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
        ArrayList<String[]> dependencyStrings = new ArrayList<String[]>();
        List<Dependency> dependencies = dependenciesFromGraph(g);
        StringBuilder sb = new StringBuilder();
        for (Dependency dep : dependencies) {
            String arr[] = new String[3];
            arr[0] = dep.relation;
            arr[1] = dep.head;
            arr[2] = dep.dependant;
            dependencyStrings.add(arr);
            //sb.append(dep.relation);
            //sb.append("(");
            //sb.append(dep.head);
            //sb.append(",");
            //sb.append(dep.dependant);
            //sb.append(")");
            //dependencyStrings.add(sb.toString());
            //sb.setLength(0);
        }
        return dependencyStrings;
    }


    private static String toDepStyle(IndexedWord fl) {
        StringBuilder buf = new StringBuilder();
        buf.append(fl.word());
        buf.append("-");
        buf.append(fl.index());
        return buf.toString();
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        CoreNLPParser c = new CoreNLPParser();
        Annotation x = c.parse("Parsing The alphabet in the Polish language contains 32 letters.", false);
        c.parseAnnotationToJson(x);
    }


}
