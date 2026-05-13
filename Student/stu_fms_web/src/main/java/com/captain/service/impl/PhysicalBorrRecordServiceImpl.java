package com.captain.service.impl;

import com.captain.entity.po.PhysicalArchives;
import com.captain.entity.po.PhysicalBorrRecord;
import com.captain.mapper.PhysicalBorrRecordMapper;
import com.captain.service.PhysicalBorrRecordService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lxb
 * @since 2020-08-29
 */
@Service
public class PhysicalBorrRecordServiceImpl extends ServiceImpl<PhysicalBorrRecordMapper, PhysicalBorrRecord> implements PhysicalBorrRecordService {

    @Autowired
    private PhysicalBorrRecordMapper physicalBorrRecordMapper;

    @Override
    public List<PhysicalBorrRecord> findOneByusernameID(Long usernameId) {
        return physicalBorrRecordMapper.findOneByusernameID(usernameId);
    }
}
