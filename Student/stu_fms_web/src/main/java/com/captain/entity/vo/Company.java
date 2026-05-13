package com.captain.entity.vo;

import lombok.Data;

/**
 * 就业单位，用于Apriori分析
 */
@Data
public class Company extends  AprioriElement{


    /**
     * 公司地址
     */
    private String address;

}
