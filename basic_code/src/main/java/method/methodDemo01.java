package method;

public class methodDemo01 {
    public static void main(String[] args) {
        int num = getSum(10,20);
        System.out.println(num);

    }
    public static  int getSum(int a, int b){
        int sum = a + b;
        return sum;

    }
}
