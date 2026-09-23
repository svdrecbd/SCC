import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import owl.ltl.*;
import owl.ltl.parser.LtlParser;
import owl.ltl.visitors.Converter;

/** A sound, incomplete search over constant and single-equation reactive policies. */
public final class PropositionalPolicySearch {
    private record Binding(int atom, Formula expression) {}
    private static long checked;
    private static long deadline;

    private static Formula replace(Formula formula, Map<Integer, Formula> assignments) {
        return formula.accept(new Converter(SyntacticFragment.ALL) {
            @Override public Formula visit(Literal literal) {
                Formula replacement = assignments.get(literal.getAtom());
                return replacement == null ? literal : literal.isNegated() ? replacement.not() : replacement;
            }
        });
    }

    private static boolean inputExpression(Formula formula, int inputCount) {
        return SyntacticFragment.SINGLE_STEP.contains(formula)
            && formula.atomicPropositions(true).length() <= inputCount;
    }

    private static Map<Integer, Formula> search(Formula formula, int first, int count,
            List<Binding> bindings, boolean desired) {
        if (count > 12) return null;
        for (Binding binding : bindings) {
            for (int assignment = 0; assignment < (1 << count); assignment++) {
                if (System.nanoTime() >= deadline) return null;
                Map<Integer, Formula> policy = new HashMap<>();
                for (int index = 0; index < count; index++) {
                    policy.put(first + index, BooleanConstant.of((assignment & (1 << index)) != 0));
                }
                if (binding != null) {
                    // Enumerate each remaining constant vector once.
                    if ((assignment & (1 << (binding.atom() - first))) != 0) continue;
                    policy.put(binding.atom(), binding.expression());
                }
                checked++;
                Formula residual = replace(formula, policy);
                if (residual.equals(BooleanConstant.of(desired))) return policy;
            }
        }
        return null;
    }

    public static void main(String[] arguments) throws Exception {
        String[] inputNames = arguments[1].split(",");
        String[] outputNames = arguments[2].split(",");
        List<String> names = new ArrayList<>(List.of(inputNames));
        names.addAll(List.of(outputNames));
        LabelledFormula labelled = LtlParser.parse(Files.readString(Path.of(arguments[0])).strip(), names);
        Formula formula = labelled.formula();
        int inputCount = inputNames.length;
        List<Binding> bindings = new ArrayList<>();
        bindings.add(null);
        // This reads the specification, not labels or published solutions.
        for (Formula subformula : new TreeSet<>(formula.subformulas(GOperator.class))) {
            Formula operand = ((GOperator) subformula).operand();
            if (!(operand instanceof Biconditional equivalence)) continue;
            for (int direction = 0; direction < 2; direction++) {
                Formula left = direction == 0 ? equivalence.leftOperand() : equivalence.rightOperand();
                Formula right = direction == 0 ? equivalence.rightOperand() : equivalence.leftOperand();
                if (left instanceof Literal literal && literal.getAtom() >= inputCount
                        && inputExpression(right, inputCount)) {
                    bindings.add(new Binding(literal.getAtom(), literal.isNegated() ? right.not() : right));
                }
            }
        }
        long started = System.nanoTime();
        deadline = started + (long)(Double.parseDouble(arguments[3]) * 1_000_000_000);
        Map<Integer, Formula> policy = search(formula, inputCount, outputNames.length, bindings, true);
        String result = "UNKNOWN";
        if (policy != null) result = "REALIZABLE";
        else if (System.nanoTime() < deadline) {
            List<Binding> constants = new ArrayList<>();
            constants.add(null);
            policy = search(formula, 0, inputCount, constants, false);
            if (policy != null) result = "UNREALIZABLE";
        }
        System.out.println("RESULT\t" + result);
        System.out.println("POLICIES_CHECKED\t" + checked);
        System.out.println("REACTIVE_BINDINGS\t" + (bindings.size() - 1));
        System.out.println("SEARCH_SECONDS\t" + ((System.nanoTime() - started) / 1_000_000_000.0));
        System.out.println("LIMIT_REACHED\t" + (System.nanoTime() >= deadline));
        if (policy != null) {
            for (int atom : new TreeSet<>(policy.keySet())) {
                System.out.println("POLICY\t" + names.get(atom) + "\t" + labelled.wrap(policy.get(atom)));
            }
            System.out.println("RESIDUAL\t" + labelled.wrap(replace(formula, policy)));
        }
    }
}
