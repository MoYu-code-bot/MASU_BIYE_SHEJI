package ifDemo;

import java.util.Scanner;

public class ifDemo06 {
    public static void main(String[] args) {
        //输入变量x,y,判断点所在的区域
        //1、原点（x=0且y=0）2、在一、二、三、四象限3、在x轴上4、在y轴上5、其他区域
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入x坐标：");
        int x = sc.nextInt();
        System.out.println("请输入y坐标：");
        int y = sc.nextInt();
        if (x == 0 && y == 0){
            System.out.println("原点");
        }else if (x > 0 && y > 0){
            System.out.println("第一象限");
        }else if (x < 0 && y > 0){
            System.out.println("第二象限");
        }else if (x < 0 && y < 0){
            System.out.println("第三象限");
        }else if (x > 0 && y < 0){
            System.out.println("第四象限");
        }else if (x == 0){
            System.out.println("在x轴上");
        }else if (y == 0){
            System.out.println("在y轴上");
        }
    }
}
