package forDemo;

import java.util.Scanner;

public class forDemo02 {
    public static void main(String[] args) {
        // S(n) = 1 -2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + ... + n
        /*
        输入：4
        说明：S(4) = 1 - 2 + 3 - 4 = -2
        输出：-2
        */
        Scanner sc = new Scanner(System.in);
        System.out.print("输入：");
        int n = sc.nextInt();

        int sum = 0;
        //用 StringBuilder 在循环中拼接表达式字符串。奇数位前拼 +，偶数位前拼 -，第一个数字前不加符号。
        StringBuilder sb = new StringBuilder();
        sb.append("S(").append(n).append(") = ");

        for (int i = 1; i <= n; i++){
            if (i % 2 == 0){
                sum -= i;
            }else {
                sum += i;
            }
            if (i == 1){
                sb.append(i);
            }else if (i % 2 == 0){
                sb.append(" - ").append(i);
            }else {
                sb.append(" + ").append(i);
            }
        }
        sb.append(" = ").append(sum);
        System.out.println("说明：" + sb);
        System.out.println("输出：" + sum);
    }
}
