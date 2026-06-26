package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.hotpot.common.Result;
import com.hotpot.entity.Reservation;
import com.hotpot.service.CustomerService;
import com.hotpot.service.ReservationService;
import com.hotpot.service.StoreService;
import com.hotpot.vo.DashboardVO;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Api(tags = "B端-数据概览")
@RestController
@RequestMapping("/admin/dashboard")
@RequiredArgsConstructor
public class AdminDashboardController {

    private final StoreService storeService;
    private final ReservationService reservationService;
    private final CustomerService customerService;

    @GetMapping("overview")
    @ApiOperation("首页数据概览")
    public Result<DashboardVO> dashboard() {
        DashboardVO vo = new DashboardVO();
        vo.setTotalStores(storeService.count());
        vo.setTotalReservations(reservationService.count());
        vo.setTotalCustomers(customerService.count());

        // 今日预订数
        LocalDateTime startOfDay = LocalDate.now().atStartOfDay();
        LocalDateTime endOfDay = startOfDay.plusDays(1);
        LambdaQueryWrapper<Reservation> wrapper = new LambdaQueryWrapper<Reservation>()
                .ge(Reservation::getCreateTime, startOfDay)
                .lt(Reservation::getCreateTime, endOfDay);
        vo.setTodayOrderCount(reservationService.count(wrapper));

        return Result.success(vo);
    }
}
