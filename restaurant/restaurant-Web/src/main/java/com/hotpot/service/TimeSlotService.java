package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.entity.TimeSlot;

import java.util.List;

public interface TimeSlotService extends IService<TimeSlot> {

    List<TimeSlot> listByStoreId(Long storeId);
}
