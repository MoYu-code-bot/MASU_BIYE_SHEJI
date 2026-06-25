package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.entity.TimeSlot;
import com.hotpot.mapper.TimeSlotMapper;
import com.hotpot.service.TimeSlotService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TimeSlotServiceImpl extends ServiceImpl<TimeSlotMapper, TimeSlot> implements TimeSlotService {

    @Override
    public List<TimeSlot> listByStoreId(Long storeId) {
        return list(new LambdaQueryWrapper<TimeSlot>()
                .eq(TimeSlot::getStoreId, storeId)
                .eq(TimeSlot::getStatus, 1)
                .orderByAsc(TimeSlot::getStartTime));
    }
}
