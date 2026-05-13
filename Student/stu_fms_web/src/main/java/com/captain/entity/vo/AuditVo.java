package com.captain.entity.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class AuditVo implements Serializable{

    //Id
    private Long id;

    private String archiveId;

    //学生Id
    private String stuId;

    //审核类型
    private String dictValue;

    //审核描叙
    private String dictDesc;

    //审核码
    private String dictCode;

    //审核状态
    private String isAudited;
}
