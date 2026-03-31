package whileDemo;

import java.util.Scanner;

public class whileDemo01 {
    public static void main(String[] args) {
        //本金：10000, 银行复利:1.7%, 多少年实现本金翻倍？
        double money = 10000;
        double exprtMoney = 20000;
        int year = 0;
        while (money < exprtMoney){
            money += money * 0.017;
            year++;
        }
        System.out.println("需要"+year+"年实现本金翻倍");
    }
}
