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
}
