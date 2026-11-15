/* Shared chart and color helpers for the workspace theme. */
(function attachWorkspaceTheme(global) {
    'use strict';

    var fallback = Object.freeze({
        canvas: '#121212',
        surface: '#1d1d1f',
        surfaceRaised: '#242426',
        text: '#f5f5f7',
        textSecondary: '#a1a1a6',
        border: 'rgba(255, 255, 255, 0.12)',
        grid: 'rgba(255, 255, 255, 0.08)',
        accent: '#0a84ff',
        accentMuted: '#64d2ff',
        positive: '#30d158',
        negative: '#ff453a',
        volume: 'rgba(10, 132, 255, 0.55)',
        crosshair: 'rgba(245, 245, 247, 0.28)'
    });

    var tokenMap = Object.freeze({
        canvas: '--ws-canvas',
        surface: '--ws-surface',
        surfaceRaised: '--ws-surface-raised',
        text: '--ws-text',
        textSecondary: '--ws-text-secondary',
        border: '--ws-border',
        grid: '--ws-chart-grid',
        accent: '--ws-chart-line',
        accentMuted: '--ws-chart-line-muted',
        positive: '--ws-chart-down',
        negative: '--ws-chart-up',
        volume: '--ws-chart-volume',
        crosshair: '--ws-chart-crosshair'
    });

    function colors() {
        var root = global.document && global.document.documentElement;
        var style = root && global.getComputedStyle ? global.getComputedStyle(root) : null;
        var result = {};
        Object.keys(fallback).forEach(function (key) {
            var value = style && style.getPropertyValue(tokenMap[key]);
            result[key] = value && value.trim() ? value.trim() : fallback[key];
        });
        return result;
    }

    function merge(target) {
        var sources = Array.prototype.slice.call(arguments, 1);
        sources.forEach(function (source) {
            if (!source || typeof source !== 'object') return;
            Object.keys(source).forEach(function (key) {
                var incoming = source[key];
                if (incoming && typeof incoming === 'object' && !Array.isArray(incoming)) {
                    var current = target[key] && typeof target[key] === 'object' ? target[key] : {};
                    target[key] = merge({}, current, incoming);
                } else {
                    target[key] = incoming;
                }
            });
        });
        return target;
    }

    function lightweightChartOptions(overrides) {
        var c = colors();
        return merge({
            layout: { background: { color: c.canvas }, textColor: c.textSecondary, fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif' },
            grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
            crosshair: { vertLine: { color: c.crosshair, labelBackgroundColor: c.surfaceRaised }, horzLine: { color: c.crosshair, labelBackgroundColor: c.surfaceRaised } },
            rightPriceScale: { borderColor: c.border },
            leftPriceScale: { borderColor: c.border },
            timeScale: { borderColor: c.border, timeVisible: true, secondsVisible: false },
            handleScroll: { mouseWheel: true, pressedMouseMove: true },
            handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true }
        }, overrides);
    }

    function candleOptions(overrides) {
        var c = colors();
        return merge({
            upColor: c.negative,
            downColor: c.positive,
            borderUpColor: c.negative,
            borderDownColor: c.positive,
            wickUpColor: c.negative,
            wickDownColor: c.positive,
            borderVisible: false
        }, overrides);
    }

    function volumeOptions(overrides) {
        var c = colors();
        return merge({
            color: c.volume,
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume',
            lastValueVisible: false,
            priceLineVisible: false
        }, overrides);
    }

    function chartJsOptions(overrides) {
        var c = colors();
        return merge({
            color: c.textSecondary,
            borderColor: c.grid,
            plugins: {
                legend: { labels: { color: c.textSecondary, usePointStyle: true, boxWidth: 8 } },
                tooltip: { backgroundColor: c.surfaceRaised, titleColor: c.text, bodyColor: c.textSecondary, borderColor: c.border, borderWidth: 1 }
            },
            scales: {
                x: { grid: { color: c.grid }, ticks: { color: c.textSecondary } },
                y: { grid: { color: c.grid }, ticks: { color: c.textSecondary } }
            }
        }, overrides);
    }

    global.WorkspaceTheme = Object.freeze({
        getColors: colors,
        merge: merge,
        lightweightChartOptions: lightweightChartOptions,
        candleOptions: candleOptions,
        volumeOptions: volumeOptions,
        chartJsOptions: chartJsOptions
    });
}(window));
