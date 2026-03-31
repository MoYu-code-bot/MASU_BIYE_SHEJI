package test;

public class LiteraDemo01 {
    public static void main(String[] args) {
        //定义一个变量用来记录微信零钱余额
        double wxmoney = 0;
        //定义一个变量用来记录支付宝零钱余额
        double alimoney = 0;
        //定义一个变量用来记录银行卡余额
        double bankmoney = 20;

        //输出现在总共有多少钱？
        System.out.println("目前总共有：" + (wxmoney + alimoney + bankmoney) + "元");

        //微信收了10元红包
        //原来的余额 + 10元红包 = 新的余额
        wxmoney = wxmoney + 10;
        System.out.println("微信收了10元红包，目前微信余额为：" + wxmoney);


    }
}
