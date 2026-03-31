package ifDemo;

import java.util.Scanner;

public class ifDemo04 {
    public static void main(String[] args) {
        //计算电费（阶梯计费原则）
        //[0 ~ 100]度，按0.5元/度
        //[100 ~ 200]度，按0.8元/度
        //[超过200]度，按1.2元/度
        //输入变量usage表示实际用电量，
        //输出总电费cost
        //示例输入：usage = 150
        //输出：cost = 100 * 0.5 + 50 * 0.8 = 90
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入实际用电量：");
        int usage = sc.nextInt();
        if (usage >= 0 && usage <= 100){
            double cost = usage * 0.5;
            System.out.println("总电费：" + cost);
        }else if (usage > 100 && usage <= 200){
            double cost =100 * 0.5 + (usage - 100) * 0.8;
            System.out.println("总电费：" + cost);
        }else if (usage > 200){
            double cost = 100 * 0.5 + 100 * 0.8 + (usage - 200) * 1.2;
            System.out.println("总电费：" + cost);
        }else {
            System.out.println("输入有误！");
        }
    }
}