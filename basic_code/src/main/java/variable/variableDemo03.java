package variable;

public class variableDemo03 {
    public static void main(String[] args) {
    //循环1~100，判断含7的数字或者7的倍数，打印"不需要该数字"
        for (int i = 1; i <= 100; i++) {
            if(i % 7 == 0 || i / 10 % 10 ==7 || i / 7 == 0 ){
                System.out.println("不需要该数字");
                continue;
            }
            System.out.println(i);
        }
    }
}
