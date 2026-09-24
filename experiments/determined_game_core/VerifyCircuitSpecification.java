import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import owl.bdd.MtBdd;
import owl.game.Game;
import owl.ltl.parser.LtlParser;
import owl.ltl.*;
import owl.ltl.visitors.Converter;
import semml.controller.Controller;
import semml.modelchecking.ModelChecker;

/** Checks the complete circuit graph against LTL with the pinned Owl backend. */
public final class VerifyCircuitSpecification implements Controller<Integer, BitSet> {
    private final List<String> inputNames;
    private final List<String> outputNames;
    private final boolean systemWins;
    private final List<List<OutputEdge<Integer, BitSet>>> transitions = new ArrayList<>();

    private static BitSet bits(int value) {
        return BitSet.valueOf(new long[] {value});
    }

    private VerifyCircuitSpecification(JsonObject record) {
        systemWins = record.get("system_wins").getAsBoolean();
        List<String> environment = new ArrayList<>();
        record.getAsJsonArray("environment_names").forEach(value -> environment.add(value.getAsString()));
        List<String> system = new ArrayList<>();
        record.getAsJsonArray("system_names").forEach(value -> system.add(value.getAsString()));
        inputNames = systemWins ? environment : system;
        outputNames = systemWins ? system : environment;
        if (inputNames.size() != record.get("input_count").getAsInt()
                || outputNames.size() != record.get("output_count").getAsInt()
                || record.get("initial").getAsInt() != 0
                || inputNames.size() + outputNames.size() > 10) {
            throw new IllegalArgumentException("Circuit interface mismatch.");
        }
        JsonArray states = record.getAsJsonArray("transitions");
        if (states.isEmpty() || states.size() > 1024) throw new IllegalArgumentException("State bound.");
        for (var state : states) {
            var row = state.getAsJsonArray();
            if (row.size() != (1 << inputNames.size())) throw new IllegalArgumentException("Not input-total.");
            List<OutputEdge<Integer, BitSet>> edges = new ArrayList<>();
            Integer environmentChoice = null;
            for (int opponent = 0; opponent < row.size(); opponent++) {
                var edge = row.get(opponent).getAsJsonObject();
                int choice = edge.get("output").getAsInt();
                int successor = edge.get("successor").getAsInt();
                if (edge.get("opponent").getAsInt() != opponent || choice < 0 || choice >= (1 << outputNames.size())
                        || successor < 0 || successor >= states.size()) throw new IllegalArgumentException("Invalid edge.");
                if (!systemWins && environmentChoice != null && choice != environmentChoice)
                    throw new IllegalArgumentException("Noncausal environment choice.");
                environmentChoice = choice;
                edges.add(new OutputEdge<>(successor, bits(choice << (systemWins ? inputNames.size() : 0))));
            }
            transitions.add(edges);
        }
    }

    public BitSet output(Integer state, BitSet input) {
        int index = input.isEmpty() ? 0 : (int) (input.toLongArray()[0] >>> (systemWins ? 0 : outputNames.size()));
        return transitions.get(state).get(index).output();
    }
    public Integer initialState() { return 0; }
    public List<Integer> orderedStates() {
        return java.util.stream.IntStream.range(0, transitions.size()).boxed().toList();
    }
    private MtBdd<OutputEdge<Integer, BitSet>> tree(int state, int variable, int assignment) {
        if (variable == inputNames.size()) return MtBdd.of(transitions.get(state).get(assignment));
        return MtBdd.of(variable + (systemWins ? 0 : outputNames.size()), tree(state, variable+1, assignment | (1 << variable)), tree(state, variable+1, assignment));
    }
    public MtBdd<OutputEdge<Integer, BitSet>> edgeTree(Integer state) { return tree(state, 0, 0); }
    public MtBdd<OutputEdge<Integer, BitSet>> edgeTreeDet(Integer state) { return edgeTree(state); }
    public Set<OutputEdge<Integer, BitSet>> edges(Integer state) { return new HashSet<>(transitions.get(state)); }
    public Game.Owner perspective() { return systemWins ? Game.Owner.PLAYER_2 : Game.Owner.PLAYER_1; }
    public List<String> inputs() { return inputNames; }
    public List<String> outputs() { return outputNames; }

    public static void main(String[] arguments) throws Exception {
        var records = new Gson().fromJson(Files.readString(Path.of(arguments[0])), JsonArray.class);
        for (var item : records) {
            var record = item.getAsJsonObject();
            long started = System.nanoTime();
            var controller = new VerifyCircuitSpecification(record);
            List<String> names = new ArrayList<>();
            record.getAsJsonArray("environment_names").forEach(value -> names.add(value.getAsString()));
            record.getAsJsonArray("system_names").forEach(value -> names.add(value.getAsString()));
            var labelled = LtlParser.parse(record.get("formula").getAsString(), names);
            boolean specialized = arguments.length > 1 && arguments[1].equals("specialize-constant-environment");
            if (specialized) {
                if (controller.systemWins || controller.transitions.size() != 1)
                    throw new IllegalArgumentException("Specialization requires a one-state environment policy.");
                BitSet choice = controller.transitions.get(0).get(0).output();
                Formula reduced = labelled.formula().accept(new Converter(SyntacticFragment.ALL) {
                    @Override public Formula visit(Literal literal) {
                        return literal.getAtom() < controller.outputNames.size()
                            ? BooleanConstant.of(choice.get(literal.getAtom()) != literal.isNegated()) : literal;
                    }
                });
                labelled = LabelledFormula.of(reduced,names);
            }
            boolean accepted = ModelChecker.check(labelled, controller);
            System.out.println(new Gson().toJson(Map.of("name",record.get("name").getAsString(), "accepted",accepted,
                "seconds", (System.nanoTime()-started)/1e9, "states",controller.transitions.size())));
        }
    }
}
