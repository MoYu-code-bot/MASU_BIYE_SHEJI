package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Reservation;
import com.hotpot.service.ReservationService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

@Api(tags = "B端-预订管理")
@RestController
@RequestMapping("/admin/reservations")
@RequiredArgsConstructor
public class AdminReservationController {

    private final ReservationService reservationService;

    @GetMapping("list")
    @ApiOperation("分页查询预订")
    public Result<PageResult<Reservation>> page(@ApiParam("分页参数") PageQuery query,
                                                @ApiParam("门店ID") @RequestParam(required = false) Long storeId,
                                                @ApiParam("预订状态") @RequestParam(required = false) Integer status,
                                                @ApiParam("预订日期") @RequestParam(required = false) @DateTimeFormat(pattern = "yyyy-MM-dd") LocalDate date) {
        Page<Reservation> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<Reservation> wrapper = new LambdaQueryWrapper<>();
        if (storeId != null) {
            wrapper.eq(Reservation::getStoreId, storeId);
        }
        if (status != null) {
            wrapper.eq(Reservation::getStatus, status);
        }
        if (date != null) {
            wrapper.eq(Reservation::getReserveDate, date);
        }
        wrapper.orderByDesc(Reservation::getCreateTime);
        return Result.success(PageResult.of(reservationService.page(page, wrapper)));
    }

    @PutMapping("confirm")
    @ApiOperation("确认预订")
    public Result<?> confirm(@ApiParam("预订ID") @RequestParam Long reservationId) {
        reservationService.confirm(reservationId);
        return Result.success();
    }

    @PutMapping("reject")
    @ApiOperation("拒绝预订")
    public Result<?> reject(@ApiParam("预订ID") @RequestParam Long reservationId) {
        reservationService.reject(reservationId);
        return Result.success();
    }

    @PutMapping("arrive")
    @ApiOperation("到店确认")
    public Result<?> arrive(@ApiParam("预订ID") @RequestParam Long reservationId) {
        reservationService.arrive(reservationId);
        return Result.success();
    }

    @PutMapping("complete")
    @ApiOperation("完成预订")
    public Result<?> complete(@ApiParam("预订ID") @RequestParam Long reservationId) {
        reservationService.complete(reservationId);
        return Result.success();
    }

    @PutMapping("noshow")
    @ApiOperation("标记未到店")
    public Result<?> noShow(@ApiParam("预订ID") @RequestParam Long reservationId) {
        reservationService.noShow(reservationId);
        return Result.success();
    }
}
