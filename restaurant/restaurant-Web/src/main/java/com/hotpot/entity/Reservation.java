package com.hotpot.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;

import java.io.Serializable;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@TableName("reservation")
public class Reservation implements Serializable {

    private static final long serialVersionUID = 1L;

    @TableId(type = IdType.AUTO)
    private Long id;

    private String orderNo;

    private Long customerId;

    private Long storeId;

    private Long tableTypeId;

    private LocalDate reserveDate;

    private Long timeSlotId;

    private Integer guestCount;

    private String customerName;

    private String customerPhone;

    private String remark;

    private Integer status;

    private String cancelReason;

    private LocalDateTime confirmTime;

    private LocalDateTime arriveTime;

    private LocalDateTime completeTime;

    @TableLogic
    private Integer deleted;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;

    @TableField(fill = FieldFill.INSERT_UPDATE)
    private LocalDateTime updateTime;

    /** 是否已评价（非数据库字段，service层填充） */
    @TableField(exist = false)
    private Boolean hasReviewed;

    /** 门店名称（非数据库字段，service层填充） */
    @TableField(exist = false)
    private String storeName;

    /** 套餐名称（非数据库字段，service层填充） */
    @TableField(exist = false)
    private String dishName;

    /** 时段文本（如"11:00-12:00"，非数据库字段，service层填充） */
    @TableField(exist = false)
    private String timeSlotText;

    /** 预订套餐ID */
    private Long dishId;
}
