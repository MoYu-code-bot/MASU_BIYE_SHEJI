package com.captain.service;

import com.captain.entity.po.PhysicalArchives;
import com.captain.entity.po.PhysicalBorrRecord;
import com.baomidou.mybatisplus.extension.service.IService;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lxb
 * @since 2020-08-29
 */
public interface PhysicalBorrRecordService extends IService<PhysicalBorrRecord> {

    List<PhysicalBorrRecord> findOneByusernameID(Long usernameId);
}
