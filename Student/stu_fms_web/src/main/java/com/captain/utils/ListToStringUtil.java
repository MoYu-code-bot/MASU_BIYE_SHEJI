package com.captain.utils;

import java.util.List;

public class ListToStringUtil {

    public static String convert(List<Long> list) {

        String s = "";

        for (int i = 0; i < list.size(); i++) {
            s += list.get(i)+",";
        }
        s = s.substring(0,s.length()-1);
        return s;
    }

//    public static void main(String[] args) {
//        List<Long> longs = new ArrayList<>();
//        longs.add(22l);
//        longs.add(23l);
//        longs.add(26l);
//        String convert = ListToStringUtil.convert(longs);
//        System.out.println(convert);
//    }
}
