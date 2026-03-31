package ifDemo;

import java.util.Scanner;

public class ifDemo02 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("请输入外卖价格：");
        int price = sc.nextInt();
        double price1 = price * 0.9;
        System.out.println("饿了么平台优惠后的价格是：" + price1);

        double price2 = 0;
        if (price >= 30){
             price2 = price - 10;
            System.out.println("美团平台优惠后的价格是：" + price2);
        }else {
            price2 = price;
            System.out.println("美团未满30元，没有优惠：" + price);
        }
        if (price1 < price2){
            System.out.println("饿了么优惠价格更优惠");
        }else {
            System.out.println("美团优惠价格更优惠");
        }
    }
}
