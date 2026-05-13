package com.captain.mapper;

import com.captain.entity.po.Audit;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Update;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
public interface AuditMapper extends BaseMapper<Audit> {

    /**
     * 更新其他表的某一条记录的审核状态
     * @param tableName
     * @param id
     * @return
     */
    @Update("update ${tableName} set audit_status_code=3 where id=#{id}")
    Integer updateOtherTableAuditStatus(String tableName,Long id);
}
