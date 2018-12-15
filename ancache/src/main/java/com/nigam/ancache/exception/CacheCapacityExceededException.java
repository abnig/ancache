package com.nigam.ancache.exception;

public class CacheCapacityExceededException extends RuntimeException {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3060557701028616456L;

	public CacheCapacityExceededException() {
		super();
	}

	public CacheCapacityExceededException(String message) {
		super(message);
	}

	public CacheCapacityExceededException(Throwable cause) {
		super(cause);
	}

	public CacheCapacityExceededException(String message, Throwable cause) {
		super(message, cause);
	}

	public CacheCapacityExceededException(String message, Throwable cause, boolean enableSuppression,
			boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

}
