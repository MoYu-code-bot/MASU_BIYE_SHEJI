package forDemo;

public class forDemo01 {
    public static void main(String[] args) {
        /*
        有一组特殊的数字，从第三项开始，每一项都是前两项的数字和，请问第10项的数字是多少？
        0 1 1 2 3 5 8 13 21 34 55...
        */
        //记录两个初始值
        int a = 0;
        int b = 1;

        //记录第三个值
        int c = 0;
        for (int i = 3; i <= 10; i++){
            c = a + b;

            a = b;
            b = c;

        }
        System.out.println(c);
    }
}
