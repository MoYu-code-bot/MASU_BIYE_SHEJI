package ifDemo;

import java.util.Scanner;

public class ifDemo03 {
    public static void main(String[] args) {
        //卡拉兹函数
        Scanner sc = new Scanner(System.in);
        System.out.print("输入：");
        int n = sc.nextInt();
        if (n % 2 != 0){
            int s1 = 3 * n + 1;
            System.out.println("说明：奇数, 3 * " + n +  " + 1 = " + s1);
            System.out.print("输出：" + s1);
        }else {
            int s2 = n / 2;
            System.out.println("说明：偶数, " + n + " / 2 = " + s2);
            System.out.print("输出：" + s2);
        }
    }
}
