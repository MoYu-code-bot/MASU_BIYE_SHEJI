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
 * @author lxb
 * @since 2020-08-28
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class PhysicalArchives extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 档案类型
     */
    private Integer archivesTypeCode;

    /**
     * 档案当前状态字典码
     */
    private Integer auditStatusCode;

    /**
     * 学号
     */
    private Long stuId;

    /**
     * 档案名称
     */
    private String archiveName;

    /**
     * 存放位置
     */
    private String position;

    /**
     * 档案说明
     */
    private String archiveDesc;

    /**
     * 档案允许被借阅分钟数
     */
    private Integer borrowTime;

    /**
     * 档案图片(多张用分号分割)
     */
    private String img;

    /**
     * 入库时间
     */
    private Date inTime;

    /**
     * 操作人员ID(如：入库人员)
     */
    private Long operatorId;


}
