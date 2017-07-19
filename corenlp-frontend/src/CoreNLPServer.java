/**
 * Created by haussmae on 26.11.2014.
 */

import java.io.*;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.logging.FileHandler;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.time.SUTime;
import edu.stanford.nlp.time.TimeAnnotations;
import edu.stanford.nlp.time.TimeExpression;
import edu.stanford.nlp.time.Timex;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;

public class CoreNLPServer {

    /**
     * Logging instance.
     */
    private static final Logger LOGGER =
            Logger.getLogger(CoreNLPServer.class.getName());

    private static final int PORT = 4434;

    public static void main(String[] args) throws Exception {

        int port = PORT;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }
        FileHandler handler = new FileHandler("corenlp-" + port + "+.log.%u.%g.txt");
        LOGGER.addHandler(handler);
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/parse", new ParseHandler());
        server.setExecutor(null); // creates a default executor
        server.start();
        LOGGER.info("Listening on port " + port);
    }

    static class ParseHandler implements HttpHandler {

        private final CoreNLPParser parser;

        public ParseHandler() throws IOException, ClassNotFoundException {
            parser = new CoreNLPParser();
        }

        public void handle(HttpExchange t) throws IOException {
            LOGGER.info("Handling request from " + t.getRemoteAddress().toString());
            long start = System.currentTimeMillis();
            String textParam = "";
            boolean performNER = false;
            if (t.getRequestMethod().toLowerCase().equals("get")) {
                textParam = getTextParamFromGET(t.getRequestURI().toString());
                performNER = getUseNERParamFromGET(t.getRequestURI().toString());
            } else {
                textParam = getTextParamFromPOST(t);
                performNER = getUseNERParamFromGET(t.getRequestURI().toString());
            }
            Annotation sentence = parser.parse(textParam, performNER);
            String json = parseAnnotationToJson(sentence);
            LOGGER.info("Returning JSON: " + json);
            t.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
            t.sendResponseHeaders(200, json.getBytes().length);
            OutputStream os = t.getResponseBody();
            os.write(json.getBytes("utf-8"));
            os.close();
            long duration = System.currentTimeMillis() - start;
            LOGGER.info("Finished handling request. Took " + duration + " ms.");
        }

        public String parseAnnotationToJson(Annotation document) {
            List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
            if (sentences.size() < 1) {
                return "{\"text\": \"\", \"words\":[], \"dependencies\":[]}";
            }
            CoreMap parsedSentence = sentences.get(0);
            String text = parsedSentence.get(CoreAnnotations.TextAnnotation.class);
            List<CoreLabel> tokens = document.get(CoreAnnotations.TokensAnnotation.class);
            Tree tree = parsedSentence.get(TreeCoreAnnotations.TreeAnnotation.class);
            List<String[]> dependencyStrings = parser.dependencyStrings(parsedSentence);
            List<CoreMap> timexAnnotations = document.get(TimeAnnotations.TimexAnnotations.class);

            StringBuffer sb = new StringBuffer();
            sb.append("{");
            // Text
            sb.append("\"text\":");
            sb.append(JSONUtil.quote(text) + ",");
            // words
            sb.append("\"words\":[");
            String comma = "";
            for (CoreLabel t : tokens) {
                sb.append(comma);
                sb.append("{");
                String token = t.get(CoreAnnotations.TextAnnotation.class);
                sb.append("\"token\":" + JSONUtil.quote(token));
                String pos = t.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                if (pos != null) {
                    sb.append(", \"pos\": " + JSONUtil.quote(pos));
                }
                String lemma = t.get(CoreAnnotations.LemmaAnnotation.class);
                if (lemma != null) {
                    sb.append(", \"lemma\": " + JSONUtil.quote(lemma));
                }
                String ner = t.get(CoreAnnotations.NamedEntityTagAnnotation.class);
                if (ner != null) {
                    sb.append(", \"ner\": " + JSONUtil.quote(ner));
                }
                String nner = t.get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class);
                if (nner != null) {
                    sb.append(", \"norm-ner\": " + JSONUtil.quote(nner));
                }
                Timex timex = t.get(TimeAnnotations.TimexAnnotation.class);
                if (timex != null) {
                    sb.append(", \"time-id\": " + JSONUtil.quote(timex.tid()));
                }

                sb.append("}");
                comma = ",";
            }
            sb.append("]");

            // Only add the constituent tree if we have one.
            sb.append(comma);
            if (tree != null) {
                sb.append("\"tree\":");
                sb.append(JSONUtil.quote(tree.toString()));
                sb.append(comma);
            }
            sb.append("\"dependencies\":[");
            comma = "";
            for (String[] arr : dependencyStrings) {
                sb.append(comma);
                sb.append("[");
                sb.append(JSONUtil.quote(arr[0]));
                sb.append(",");
                sb.append(JSONUtil.quote(arr[1]));
                sb.append(",");
                sb.append(JSONUtil.quote(arr[2]));
                sb.append("]");
                comma = ",";
            }
            sb.append("]");
            sb.append("}");

            return sb.toString();
        }

        public static boolean getUseNERParamFromGET(String requestURI) throws UnsupportedEncodingException {
            int start = requestURI.indexOf("?performNER=");
            if (start == -1) {
                start = requestURI.indexOf("&performNER=");
            }
            if (start == -1) {
                return false;
            }
            return true;
        }

        public static String getTextParamFromGET(String requestURI) throws UnsupportedEncodingException {
            int start = requestURI.indexOf("?text=");
            if (start == -1) {
                start = requestURI.indexOf("&text=");
            }
            if (start == -1) {
                return "No text parameter found.";
            }
            int end = requestURI.indexOf("&", start);
            if (end == -1) {
                return URLDecoder.decode(requestURI.substring(start + 6), "utf-8");
            } else {
                return URLDecoder.decode(requestURI.substring(start + 6, end), "utf-8");
            }
        }

        public static String getTextParamFromPOST(HttpExchange t) throws IOException {
            Map<String, Object> parameters =
                    (Map<String, Object>) t.getAttribute("parameters");
            InputStreamReader isr =
                    new InputStreamReader(t.getRequestBody(), "utf-8");
            BufferedReader br = new BufferedReader(isr);
            String body = br.readLine();
            int start = body.indexOf("text=");
            if (start == -1) {
                return "No text parameter found.";
            }
            int end = body.indexOf("&", start);
            if (end == -1) {
                return URLDecoder.decode(body.substring(start + 5), "utf-8");
            } else {
                return URLDecoder.decode(body.substring(start + 5, end), "utf-8");
            }
        }
    }
    // Good JSON.
    // {"sentences": [{"parsetree": "(ROOT (NP (NN barack) (NN obama)))", "text": "barack obama", "dependencies": [["root", "ROOT", "obama"], ["nn", "obama", "barack"]], "indexeddependencies": [["root", "ROOT-0", "obama-2"], ["nn", "obama-2", "barack-1"]], "words": [["barack", {"NamedEntityTag": "PERSON", "CharacterOffsetEnd": "6", "Lemma": "barack", "PartOfSpeech": "NN", "CharacterOffsetBegin": "0"}], ["obama", {"NamedEntityTag": "PERSON", "CharacterOffsetEnd": "12", "Lemma": "obama", "PartOfSpeech": "NN", "CharacterOffsetBegin": "7"}]]}]}
}
