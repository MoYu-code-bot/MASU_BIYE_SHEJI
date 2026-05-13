package com.captain.entity.po;

import java.util.Date;
import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lll
 * @since 2020-08-28
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Employ extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 档案类型
     */
    private Long archivesTypeCode;

    /**
     * 档案当前状态字典码
     */
    private Long auditStatusCode;

    /**
     * 学号
     */
    private String stuId;

    /**
     * 毕业后入职的第一家公司名字
     */
    private String company;

    /**
     * 入职时间
     */
    private Date inTime;

    /**
     * 薪资
     */
    private Float salary;

    /**
     * 公司地址
     */
    private String address;


}
