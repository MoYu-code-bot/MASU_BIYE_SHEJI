package method;

import java.util.Scanner;

public class methodDemo06 {
    public static void main(String[] args) {
      /*  计算快递邮费
        某快递公司的运费规则如下（首重 1kg，超出部分按 kg 计算，不足 1kg 按 1kg 算）：
        1.首重 1kg：10 元；
        2.超出 1-5kg：每 kg 加 2 元；
        3.超出 5kg 以上：每 kg 加 1.5 元。
        键盘录入小数，表示用户快递的重量，计算最终的结果

        要求：
        1.快递重量必须大于 0，否则重新输入
        2.不同价位的计算，单独定义一个方法*/
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入快递的重量：");
        double weight = sc.nextDouble();
        if (weight < 0){
            System.out.println("输入的数字有误，请重新输入：");
            weight = sc.nextDouble();
        }
        double price = getPrice(weight);
        System.out.println("邮费为：" + price + "元");
    }
    public static double getPrice(double  weight){
        double price = 10;
        if(weight <= 1){
            price = 10;
        }else if (weight > 1 && weight <= 5){
            price += (weight - 1) * 2;
        }else {
            price += 5 * 2 + (weight - 5) * 1.5;
        }
        return price;

    }
}
