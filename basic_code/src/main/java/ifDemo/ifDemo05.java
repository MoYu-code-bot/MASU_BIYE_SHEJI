package ifDemo;

import java.util.Scanner;

public class ifDemo05 {
    public static void main(String[] args) {
        //充卡赠送
        //充值金额：1000，赠送金额：200，门票39元/次
        //充值金额：2000，赠送金额：500，门票35元/次
        //充值金额：3000，赠送金额：700，门票35元/次
        //充值金额：5000，赠送金额：1300，门票35元/次
        //充值金额：10000，赠送金额：2500，门票30元/次（送专属更衣箱一个）
        //充值金额：20000，赠送金额：6000，门票30元/次（送专属更衣箱一个）
        //充值金额：50000，赠送金额：15000，门票30元/次（送专属更衣箱一个和浴服一套）

        Scanner sc = new Scanner(System.in);
        System.out.println("请输入充值金额：");
        int price = sc.nextInt();
        if (price == 1000 ){
            System.out.println("充值金额：" + price + "，赠送金额：" + 200 + "，门票39元/次");
        }else if (price == 2000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 500 + "，门票35元/次");
        }else if (price == 3000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 700 + "，门票35元/次");
        }else if (price == 5000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 1300 + "，门票35元/次");
        }else if (price == 10000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 2500 + "，门票30元/次," + "赠送您专属更衣箱一个!");
        }else if (price == 20000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 6000 + "，门票30元/次," + "赠送您专属更衣箱一个!");
        }else if (price == 50000){
            System.out.println("充值金额：" + price + "，赠送金额：" + 15000 + "，门票30元/次," + "赠送您专属更衣箱一个和浴服一套!");
        }else {
            System.out.println("充值金额有误！");
        }
    }
}
