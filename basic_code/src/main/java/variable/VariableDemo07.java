package variable;

public class VariableDemo07 {
    public static void main(String[] args) {
        for (int i = 1; i <= 5; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print("*");
            }
            System.out.println();
        }
        /*for (int n = 4; n >= 0; n--) {
            for (int a = 0; a < n; a++) {
                System.out.print("*");
            }
            System.out.println();
        }*/

    }
}
